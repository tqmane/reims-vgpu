#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>

namespace ReimsABI {
template <typename T> inline T read(const void *data, size_t offset) {
    T value;
    memcpy(&value, static_cast<const uint8_t *>(data) + offset, sizeof(value));
    return value;
}
template <typename T> inline void write(void *data, size_t offset, T value) {
    memcpy(static_cast<uint8_t *>(data) + offset, &value, sizeof(value));
}

// These record layouts are measured at the 64.4.7/15.0.0 user-client boundary.
inline bool resource(const void *input, size_t size, uint8_t *legacy, size_t capacity) {
    if (!input || !legacy || size < 112 || capacity < 16 || size > capacity - 16) return false;
    const auto source = static_cast<const uint8_t *>(input);
    uint32_t kind = read<uint32_t>(source, 0);
    uint32_t field30 = read<uint32_t>(source, 0x30);
    uint32_t field58 = read<uint32_t>(source, 0x58);
    bool texture = size >= 216 && kind == 0 && field30 == 0 && field58 == 2 &&
        read<uint32_t>(source, 0x60) == size - 112;
    bool surface = size == 160 && kind == 0x82 && field30 == 0 && field58 == 6;
    bool buffer = size == 112 &&
        ((kind == 0x80 && field30 == 1 && field58 == 1) ||
         (kind == 0 && field30 <= 1 && (field58 == 0 || field58 == 1 || field58 == 5)));
    if (!buffer && !texture && !surface) return false;
    memset(legacy, 0, size + 16);
    memcpy(legacy, source, 16);
    memcpy(legacy + 0x18, source + 0x48, 8);
    memcpy(legacy + 0x20, source + 0x10, 32);
    memcpy(legacy + 0x40, source + 0x38, 16);
    memcpy(legacy + 0x58, source + 0x48, 8);
    memcpy(legacy + 0x60, source + 0x50, size - 0x50);
    if (texture) memcpy(legacy + 0x18, source + 0x98, 8);
    if (field58 == 0) write<uint64_t>(legacy, 0x18, 1);
    if (field58 == 5 || surface) write<uint32_t>(legacy, 0x68, field58 - 1);
    return true;
}
template <size_t N> inline bool resource(const void *input, size_t size, uint8_t (&legacy)[N]) {
    return resource(input, size, legacy, N);
}
}
