#include "ResourceABI.hpp"
#include "SubmissionABI.hpp"
#include "DirtyABI.hpp"
#include <assert.h>
#include <initializer_list>

static void checkSubmission() {
    uint8_t source[0x48 + 128] {}, commands[0xc0] {}, output[0x28 + 7 * 0x58] {};
    ReimsABI::write<uint32_t>(source, 8, 1);
    ReimsABI::write<uint32_t>(source, 0x0c, 0x40000001);
    ReimsABI::write<uint32_t>(source, 0x14, 0xac);
    ReimsABI::write<uint32_t>(source, 0x20, 1);
    ReimsABI::write<uint32_t>(source, 0x24, 0x80000000U | (48 + 128));
    ReimsABI::write<uint32_t>(source, 0x30, 0xac);
    ReimsABI::write<uint32_t>(source, 0x34, 0xc0);
    ReimsABI::write<uint32_t>(source, 0x40, 7);
    ReimsABI::write<uint32_t>(source, 0x44, 2);
    ReimsABI::write<uint32_t>(commands, 0, 15);
    ReimsABI::write<uint32_t>(commands, 4, 0xac);
    ReimsABI::write<uint32_t>(commands, 0xac, 0x10000);
    ReimsABI::write<uint32_t>(commands, 0xb0, 20);
    for (unsigned i = 0; i < 7; i++) {
        auto chunk = source + 0x48 + (i / 6) * 64;
        ReimsABI::write<uint32_t>(chunk, (i % 6) * 4, i + 1);
        ReimsABI::write<uint32_t>(chunk, 24 + (i % 6) * 4, i + 1);
        ReimsABI::write<uint16_t>(chunk, 48 + (i % 6) * 2, i == 6 ? 5 : 3);
        ReimsABI::write<uint32_t>(chunk, 60, (i < 6 ? 6 : 1) << 16);
    }
    auto lookup = [](void *, uint32_t id, uint64_t &length) { length = id * 1024; return id && id <= 7; };
    assert(ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output), nullptr, lookup));
    assert(ReimsABI::read<uint32_t>(output, 0x20) == 7);
    assert(ReimsABI::read<uint64_t>(output, 0x28 + 6 * 0x58 + 0x18) == 7168);
    assert(ReimsABI::read<uint32_t>(output, 0x28 + 6 * 0x58 + 0x48) == 5);
    ReimsABI::write<uint16_t>(source, 0x48 + 60, 0xa5a5);
    ReimsABI::write<uint16_t>(source, 0x48 + 64 + 60, 0x5a5a);
    assert(ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output), nullptr, lookup));
    ReimsABI::write<uint16_t>(source, 0x48 + 64 + 62, 2);
    assert(!ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output), nullptr, lookup));
    ReimsABI::write<uint16_t>(source, 0x48 + 64 + 62, 1);
    auto unaligned = [](void *context, uint32_t id, uint64_t &length) {
        length = id == 1 ? *static_cast<uint64_t *>(context) : id * 1024;
        return id && id <= 7;
    };
    for (uint64_t length : {1ULL, 1023ULL, 1024ULL, 1025ULL, 29824ULL, 952512ULL}) {
        uint32_t units = static_cast<uint32_t>((length + 1023) >> 10);
        ReimsABI::write<uint32_t>(source, 0x48 + 24, units);
        assert(ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output), &length, unaligned));
        assert(ReimsABI::read<uint64_t>(output, 0x28 + 0x18) == length);
        for (uint32_t wrong : {units - 1, units + 1}) {
            ReimsABI::write<uint32_t>(source, 0x48 + 24, wrong);
            assert(!ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output), &length, unaligned));
        }
    }
    uint64_t tooLarge = UINT64_MAX;
    ReimsABI::write<uint32_t>(source, 0x48 + 24, UINT32_MAX);
    assert(!ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output), &tooLarge, unaligned));
    ReimsABI::write<uint32_t>(source, 0x48 + 24, 1);
    for (size_t length = 0; length < sizeof(source); length++)
        assert(!ReimsABI::submissionSize(source, length, commands, sizeof(commands)));
    assert(!ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output)-1, nullptr, lookup));
    source[0x48 + 48] = 8;
    assert(!ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output), nullptr, lookup));
    source[0x48 + 48] = 3;
    source[0x48 + 64] = 99;
    assert(!ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output), nullptr, lookup));
    ReimsABI::write<uint32_t>(source, 0x40, UINT32_MAX);
    assert(!ReimsABI::submissionSize(source, sizeof(source), commands, sizeof(commands)));

    uint8_t multi[0xe8] {}, multiCommands[0xd4] {}, multiOutput[240] {};
    memcpy(multi, source, 0x28);
    memcpy(multiCommands, commands, sizeof(commands));
    ReimsABI::write<uint32_t>(multi, 0x20, 2);
    ReimsABI::write<uint32_t>(multi, 0x24, 0x800000d0);
    ReimsABI::write<uint32_t>(multiCommands, 0xc0, 0x10000);
    ReimsABI::write<uint32_t>(multiCommands, 0xc4, 20);
    for (unsigned group = 0; group < 2; group++) {
        auto record = multi + 0x28 + group * 96;
        ReimsABI::write<uint32_t>(record, 8, 0xac + group * 20);
        ReimsABI::write<uint32_t>(record, 12, 0xc0 + group * 20);
        ReimsABI::write<uint32_t>(record, 24, 1);
        ReimsABI::write<uint32_t>(record, 28, 1);
        ReimsABI::write<uint32_t>(record, 32, group + 1);
        ReimsABI::write<uint32_t>(record, 56, group + 1);
        ReimsABI::write<uint16_t>(record, 80, group ? 5 : 3);
        ReimsABI::write<uint32_t>(record, 92, 0x10000);
    }
    assert(ReimsABI::submission(multi, sizeof(multi), multiCommands, sizeof(multiCommands), multiOutput, sizeof(multiOutput), nullptr, lookup));
    assert(ReimsABI::read<uint32_t>(multiOutput, 12) == 0x800000f0);
    assert(ReimsABI::read<uint32_t>(multiOutput, 16 + 112 + 8) == 0xc0);
    assert(ReimsABI::read<uint64_t>(multiOutput, 16 + 112 + 24 + 24) == 2048);
    multi[0x90] = 0xac;
    assert(!ReimsABI::submissionSize(multi, sizeof(multi), multiCommands, sizeof(multiCommands)));
}

static void checkEvents() {
    uint8_t source[160] {}, commands[216] {}, convertedCommands[216] {}, output[152] {};
    ReimsABI::write<uint32_t>(source, 8, 1);
    ReimsABI::write<uint32_t>(source, 12, 0x40000001);
    ReimsABI::write<uint32_t>(source, 20, 172);
    ReimsABI::write<uint32_t>(source, 32, 1);
    ReimsABI::write<uint32_t>(source, 36, 112);
    ReimsABI::write<uint32_t>(source, 48, 172);
    ReimsABI::write<uint32_t>(source, 52, 192);
    ReimsABI::write<uint32_t>(source, 64, 1);
    ReimsABI::write<uint32_t>(source, 68, 1);
    ReimsABI::write<uint32_t>(source, 72, 7);
    ReimsABI::write<uint32_t>(source, 96, 7);
    ReimsABI::write<uint16_t>(source, 120, 3);
    ReimsABI::write<uint32_t>(source, 132, 0x10000);
    ReimsABI::write<uint32_t>(source, 144, 1);
    ReimsABI::write<uint32_t>(source, 148, 0xc0000001);
    ReimsABI::write<uint32_t>(source, 152, 192);
    ReimsABI::write<uint32_t>(source, 156, 216);
    ReimsABI::write<uint32_t>(commands, 0, 15);
    ReimsABI::write<uint32_t>(commands, 4, 172);
    ReimsABI::write<uint32_t>(commands, 172, 0x10000);
    ReimsABI::write<uint32_t>(commands, 176, 20);
    ReimsABI::write<uint32_t>(commands, 192, 3);
    ReimsABI::write<uint32_t>(commands, 196, 24);
    ReimsABI::write<uint32_t>(commands, 200, 0x1234);
    ReimsABI::write<uint64_t>(commands, 208, 0x100000001ULL);
    auto lookup = [](void *, uint32_t id, uint64_t &length) { length = 7168; return id == 7; };
    memcpy(convertedCommands, commands, sizeof(commands));
    assert(ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output),
        nullptr, lookup, convertedCommands));
    assert(ReimsABI::read<uint32_t>(output, 12) == 128);
    assert(ReimsABI::read<uint32_t>(output, 140) == 0xc0000001);
    assert(ReimsABI::read<uint32_t>(output, 144) == 192);
    assert(ReimsABI::read<uint32_t>(convertedCommands, 192) == 4);
    assert(ReimsABI::read<uint64_t>(convertedCommands, 208) == 0x100000001ULL);
    assert(ReimsABI::read<uint32_t>(commands, 192) == 3);
    assert(!ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output), nullptr, lookup));
    for (size_t size = 0; size < sizeof(source); size++)
        assert(!ReimsABI::submissionSize(source, size, commands, sizeof(commands)));
    for (size_t size = 0; size < sizeof(commands); size++)
        assert(!ReimsABI::submissionSize(source, sizeof(source), commands, size));
    ReimsABI::write<uint32_t>(commands, 192, 99);
    assert(!ReimsABI::submissionSize(source, sizeof(source), commands, sizeof(commands)));
    ReimsABI::write<uint32_t>(commands, 192, 4);
    ReimsABI::write<uint32_t>(commands, 204, UINT32_MAX);
    memcpy(convertedCommands, commands, sizeof(commands));
    assert(ReimsABI::submission(source, sizeof(source), commands, sizeof(commands), output, sizeof(output),
        nullptr, lookup, convertedCommands));
    assert(ReimsABI::read<uint32_t>(convertedCommands, 192) == 5);
    assert(ReimsABI::read<uint32_t>(convertedCommands, 204) == 0);
    ReimsABI::write<uint32_t>(commands, 204, 10);
    assert(!ReimsABI::submissionSize(source, sizeof(source), commands, sizeof(commands)));

    // A wait can share the initial kernel range with the removable preamble.
    uint8_t waitSource[136] {};
    memcpy(waitSource, source, sizeof(waitSource));
    ReimsABI::write<uint32_t>(waitSource, 20, 196);
    ReimsABI::write<uint32_t>(waitSource, 36, 0x80000070);
    ReimsABI::write<uint32_t>(waitSource, 48, 196);
    ReimsABI::write<uint32_t>(waitSource, 52, 216);
    memset(commands + 172, 0, 44);
    ReimsABI::write<uint32_t>(commands, 172, 4);
    ReimsABI::write<uint32_t>(commands, 176, 24);
    ReimsABI::write<uint32_t>(commands, 180, 0x1234);
    ReimsABI::write<uint32_t>(commands, 184, UINT32_MAX);
    ReimsABI::write<uint64_t>(commands, 188, 1);
    ReimsABI::write<uint32_t>(commands, 196, 0x10000);
    ReimsABI::write<uint32_t>(commands, 200, 20);
    memcpy(convertedCommands, commands, sizeof(commands));
    assert(ReimsABI::submission(waitSource, sizeof(waitSource), commands, sizeof(commands), output, sizeof(output),
        nullptr, lookup, convertedCommands));
    assert(ReimsABI::read<uint32_t>(output, 16) == 172);
    assert(ReimsABI::read<uint32_t>(output, 20) == 196);
    assert(ReimsABI::read<uint32_t>(output, 36) == 0x80000080);
    assert(ReimsABI::read<uint32_t>(convertedCommands, 172) == 5);
    ReimsABI::write<uint32_t>(waitSource, 8, UINT32_MAX);
    assert(!ReimsABI::submissionSize(waitSource, sizeof(waitSource), commands, sizeof(commands)));

    uint8_t emptySource[24] {}, emptyCommands[172] {}, emptyOutput[16] {};
    ReimsABI::write<uint32_t>(emptySource, 8, 1);
    ReimsABI::write<uint32_t>(emptySource, 12, 0xc0000001);
    ReimsABI::write<uint32_t>(emptySource, 20, 172);
    ReimsABI::write<uint32_t>(emptyCommands, 0, 15);
    ReimsABI::write<uint32_t>(emptyCommands, 4, 172);
    assert(ReimsABI::submission(emptySource, sizeof(emptySource), emptyCommands, sizeof(emptyCommands),
        emptyOutput, sizeof(emptyOutput), nullptr, lookup));
    assert(ReimsABI::read<uint32_t>(emptyOutput, 8) == 0);
    assert(ReimsABI::read<uint32_t>(emptyOutput, 12) == 0x80000010);
    emptyCommands[8] = 1;
    assert(!ReimsABI::submissionSize(emptySource, sizeof(emptySource), emptyCommands, sizeof(emptyCommands)));
}

int main() {
    assert(ReimsABI::batchSize(1, 48, 48) == 32);
    assert(ReimsABI::batchSize(64, 48, 3072) == 1544);
    assert(!ReimsABI::batchSize(0, 48, 0));
    assert(!ReimsABI::batchSize(2, 48, 48));
    assert(!ReimsABI::batchSize(2, 48, 97));
    assert(!ReimsABI::batchSize(2, 64, 128));
    assert(!ReimsABI::batchSize(UINT64_MAX, 48, UINT64_MAX));
    checkSubmission();
    checkEvents();
    uint8_t dirty[32];
    assert(ReimsABI::dirtyTexture(7, 0, 0, dirty));
    assert(ReimsABI::read<uint32_t>(dirty, 4) == 2 && ReimsABI::read<uint32_t>(dirty, 12) == 1);
    assert(ReimsABI::dirtyTexture(7, 31, 5, dirty));
    assert(ReimsABI::read<uint32_t>(dirty, 8) == 5 && ReimsABI::read<uint32_t>(dirty, 12) == 0x80000000U);
    assert(!ReimsABI::dirtyTexture(7, 32, 0, dirty));
    assert(!ReimsABI::dirtyTexture(7, 0, 0x100000000ULL, dirty));
    assert(ReimsABI::dirtyBuffer(7, 0, 0, dirty) == 0);
    assert(ReimsABI::dirtyBuffer(7, 4, 20, dirty) == 1);
    assert(ReimsABI::read<uint32_t>(dirty, 4) == 1 && ReimsABI::read<uint32_t>(dirty, 12) == 20);
    assert(ReimsABI::dirtyBuffer(7, 0x100000000ULL, 24, dirty) == 2);
    assert(ReimsABI::read<uint64_t>(dirty, 8) == 0x100000000ULL);
    assert(ReimsABI::read<uint32_t>(dirty, 16) == 7 && ReimsABI::read<uint32_t>(dirty, 20) == 4);
    assert(ReimsABI::dirtyBuffer(7, 4, 0x100000000ULL, dirty) == 2);
    assert(ReimsABI::read<uint64_t>(dirty, 24) == 0x100000000ULL);
    uint8_t input[216] {}, output[232] {};
    assert(!ReimsABI::resource(nullptr, 112, output));
    for (size_t n = 0; n <= sizeof(input); n++) {
        if (n != 112) assert(!ReimsABI::resource(input, n, output));
    }
    ReimsABI::write<uint64_t>(input, 0x10, 0x12345678000ULL);
    ReimsABI::write<uint64_t>(input, 0x48, 4096);
    ReimsABI::write<uint32_t>(input, 0x58, 1);
    assert(ReimsABI::resource(input, 112, output));
    assert(ReimsABI::read<uint64_t>(output, 0x20) == 0x12345678000ULL);
    assert(ReimsABI::read<uint64_t>(output, 0x18) == 4096);
    assert(ReimsABI::read<uint32_t>(output, 0x68) == 1);
    for (unsigned kind : {0U, 0x80U}) {
        ReimsABI::write<uint32_t>(input, 0, kind);
        ReimsABI::write<uint32_t>(input, 0x30, kind == 0x80);
        assert(ReimsABI::resource(input, 112, output));
    }
    ReimsABI::write<uint32_t>(input, 0, 0);
    ReimsABI::write<uint32_t>(input, 0x30, 0);
    ReimsABI::write<uint32_t>(input, 0x58, 0);
    assert(ReimsABI::resource(input, 112, output));
    assert(ReimsABI::read<uint64_t>(output, 0x18) == 1);
    ReimsABI::write<uint32_t>(input, 0x58, 5);
    assert(ReimsABI::resource(input, 112, output));
    assert(ReimsABI::read<uint32_t>(output, 0x68) == 4);
    ReimsABI::write<uint32_t>(input, 0x58, 2);
    ReimsABI::write<uint32_t>(input, 0x60, 104);
    ReimsABI::write<uint64_t>(input, 0x98, 256);
    input[215] = 0xa5;
    assert(ReimsABI::resource(input, 216, output));
    assert(ReimsABI::read<uint64_t>(output, 0x18) == 256 && output[231] == 0xa5);
    ReimsABI::write<uint32_t>(input, 0, 0x82);
    ReimsABI::write<uint32_t>(input, 0x58, 6);
    input[159] = 0x7b;
    assert(ReimsABI::resource(input, 160, output));
    assert(ReimsABI::read<uint32_t>(output, 0x68) == 5 && output[175] == 0x7b);
    ReimsABI::write<uint32_t>(input, 0, 0xffffffff);
    assert(!ReimsABI::resource(input, 160, output));
    uint8_t mipmapped[468] {}, converted[484] {};
    ReimsABI::write<uint32_t>(mipmapped, 0x58, 2);
    ReimsABI::write<uint32_t>(mipmapped, 0x60, 356);
    mipmapped[467] = 0x6b;
    assert(ReimsABI::resource(mipmapped, sizeof(mipmapped), converted));
    assert(converted[483] == 0x6b);
    assert(!ReimsABI::resource(mipmapped, sizeof(mipmapped), output));
    ReimsABI::write<uint32_t>(mipmapped, 0x60, 357);
    assert(!ReimsABI::resource(mipmapped, sizeof(mipmapped), converted));
}
