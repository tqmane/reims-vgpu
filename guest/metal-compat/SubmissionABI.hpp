#pragma once
#include "ResourceABI.hpp"

namespace ReimsABI {
using ResourceLookup = bool (*)(void *, uint32_t, uint64_t &);

inline size_t batchSize(uint64_t count, uint64_t entrySize, uint64_t bytes) {
    if (!count || entrySize != 48 || count > (UINT32_MAX - 8) / 24 ||
        count > SIZE_MAX / 48 || bytes != count * 48) return 0;
    return 8 + static_cast<size_t>(count) * 24;
}

// The authored event probe measures shared-event signal 3 -> 4 and wait 4 -> 5.
// shortcut: finite shared-event waits and other kernel commands remain unsupported.
inline size_t walkSubmission(const uint8_t *source, size_t sourceSize,
                            const uint8_t *commands, size_t commandSize,
                            uint8_t *output, uint8_t *legacyCommands,
                            void *context, ResourceLookup lookup) {
    if (!source || !commands || sourceSize < 24 || commandSize < 0xac ||
        read<uint32_t>(commands, 0) != 15 || read<uint32_t>(commands, 4) != 0xac)
        return 0;
    for (size_t i = 8; i < 0xac; i++) if (commands[i]) return 0;
    size_t inputCursor = 0, outputCursor = 0;
    uint32_t previousEnd = 0;
    for (;;) {
        if (inputCursor > sourceSize || sourceSize - inputCursor < 16) return 0;
        const uint8_t *header = source + inputCursor;
        uint32_t count = read<uint32_t>(header, 8), tag = read<uint32_t>(header, 12);
        size_t inputLength = 0, outputLength = 16;
        if (tag & 0x40000000U) {
            if ((tag & 0x7fffffffU) != 0x40000001U || !count ||
                count > (sourceSize - inputCursor - 16) / 8) return 0;
            inputLength = 16 + static_cast<size_t>(count) * 8;
            uint32_t emitted = 0;
            for (uint32_t i = 0; i < count; i++) {
                uint32_t begin = read<uint32_t>(header, 16 + i * 8);
                uint32_t end = read<uint32_t>(header, 20 + i * 8);
                if (begin != previousEnd || end <= begin || end > commandSize) return 0;
                uint32_t cursor = begin;
                if (cursor == 0) {
                    if (end < 0xac) return 0;
                    cursor = begin = 0xac;
                }
                while (cursor < end) {
                    if (end - cursor < 24) return 0;
                    if (output && !legacyCommands) return 0;
                    uint32_t op = read<uint32_t>(commands, cursor);
                    uint32_t field0c = read<uint32_t>(commands, cursor + 12);
                    if ((op != 3 && op != 4) || read<uint32_t>(commands, cursor + 4) != 24 ||
                        !read<uint32_t>(commands, cursor + 8) ||
                        (op == 3 ? field0c != 0 : field0c != UINT32_MAX)) return 0;
                    if (legacyCommands) {
                        write<uint32_t>(legacyCommands, cursor, op + 1);
                        write<uint32_t>(legacyCommands, cursor + 12, 0);
                    }
                    cursor += 24;
                }
                if (begin != end) {
                    if (output) {
                        write<uint32_t>(output, outputCursor + 16 + emitted * 8, begin);
                        write<uint32_t>(output, outputCursor + 20 + emitted * 8, end);
                    }
                    emitted++;
                }
                previousEnd = end;
            }
            outputLength = emitted ? 16 + static_cast<size_t>(emitted) * 8 : 0;
            if (outputLength && output) {
                memcpy(output + outputCursor, header, 16);
                write<uint32_t>(output, outputCursor + 8, emitted);
            }
            if (!emitted && !outputCursor && (tag & 0x80000000U)) {
                // Empty Metal buffers still submit an empty legacy segment and receive callbacks.
                outputLength = 16;
                if (output) {
                    memcpy(output, header, 8);
                    write<uint32_t>(output, 8, 0);
                    write<uint32_t>(output, 12, 0x80000010);
                }
            }
        } else {
            inputLength = tag & 0x3fffffffU;
            if (inputLength < 16 || inputLength > sourceSize - inputCursor || !count ||
                count > (inputLength - 16) / 32) return 0;
            size_t cursor = 16;
            for (uint32_t group = 0; group < count; group++) {
                if (cursor > inputLength || inputLength - cursor < 32) return 0;
                const uint8_t *record = header + cursor;
                uint32_t begin = read<uint32_t>(record, 8), end = read<uint32_t>(record, 12);
                uint32_t resources = read<uint32_t>(record, 24), chunks = read<uint32_t>(record, 28);
                if (begin != previousEnd || end <= begin || end > commandSize || (end - begin) % 20 ||
                    !resources || resources > (0x3fffffffU - 24) / 88 || chunks != (resources + 5) / 6 ||
                    chunks > (inputLength - cursor - 32) / 64) return 0;
                for (uint32_t at = begin; at < end; at += 20)
                    if (read<uint32_t>(commands, at) != 0x10000 || read<uint32_t>(commands, at + 4) != 20)
                        return 0;
                size_t recordSize = 24 + static_cast<size_t>(resources) * 88;
                if (outputLength > 0x3fffffffU - recordSize) return 0;
                if (output) {
                    uint8_t *destination = output + outputCursor + outputLength;
                    memcpy(destination, record, 16);
                    write<uint32_t>(destination, 16, resources);
                    write<uint32_t>(destination, 20, resources);
                    for (uint32_t i = 0; i < resources; i++) {
                        const uint8_t *chunk = record + 32 + (i / 6) * 64;
                        uint32_t remaining = resources - (i / 6) * 6;
                        // The producer writes a 16-bit count at 0x3e; 0x3c is unused padding.
                        if (read<uint16_t>(chunk, 62) != (remaining < 6 ? remaining : 6)) return 0;
                        uint32_t id = read<uint32_t>(chunk, (i % 6) * 4);
                        uint32_t kibibytes = read<uint32_t>(chunk, 24 + (i % 6) * 4);
                        uint16_t access = read<uint16_t>(chunk, 48 + (i % 6) * 2);
                        uint64_t length = 0;
                        // New records round up to KiB; legacy records carry exact byte lengths.
                        if (!(access & 1) || (access & ~7) || !lookup(context, id, length) ||
                            (length >> 10) + ((length & 1023) != 0) != kibibytes) return 0;
                        uint8_t *entry = destination + 24 + i * 88;
                        write<uint32_t>(entry, 0, id);
                        write<uint64_t>(entry, 0x18, length);
                        write<uint32_t>(entry, 0x48, access);
                        write<uint32_t>(entry, 0x54, 0x1ffff);
                    }
                }
                outputLength += recordSize;
                cursor += 32 + static_cast<size_t>(chunks) * 64;
                previousEnd = end;
            }
            if (cursor != inputLength) return 0;
            if (output) {
                memcpy(output + outputCursor, header, 16);
                write<uint32_t>(output, outputCursor + 12,
                    (tag & 0x80000000U) | static_cast<uint32_t>(outputLength));
            }
        }
        if (outputLength > UINT32_MAX - outputCursor) return 0;
        outputCursor += outputLength;
        if (tag & 0x80000000U) return outputCursor;
        inputCursor += inputLength;
    }
}

inline size_t submissionSize(const uint8_t *source, size_t sourceSize,
                             const uint8_t *commands, size_t commandSize) {
    return walkSubmission(source, sourceSize, commands, commandSize, nullptr, nullptr, nullptr, nullptr);
}

inline bool submission(const uint8_t *source, size_t sourceSize,
                       const uint8_t *commands, size_t commandSize,
                       uint8_t *output, size_t outputSize, void *context, ResourceLookup lookup,
                       uint8_t *legacyCommands = nullptr) {
    size_t size = submissionSize(source, sourceSize, commands, commandSize);
    if (!size || !output || outputSize < size || !lookup) return false;
    memset(output, 0, size);
    return walkSubmission(source, sourceSize, commands, commandSize,
        output, legacyCommands, context, lookup) == size;
}
}
