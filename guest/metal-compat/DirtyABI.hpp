#pragma once
#include "ResourceABI.hpp"

namespace ReimsABI {
inline bool dirtyTexture(uint32_t identifier, uint64_t level, uint64_t slice,
                         uint8_t (&output)[32]) {
    if (level >= 32 || slice > UINT32_MAX) return false;
    memset(output, 0, sizeof(output));
    write<uint32_t>(output, 0, identifier);
    write<uint32_t>(output, 4, 2);
    write<uint32_t>(output, 8, static_cast<uint32_t>(slice));
    write<uint32_t>(output, 12, 1U << level);
    return true;
}

inline unsigned dirtyBuffer(uint32_t identifier, uint64_t offset, uint64_t length,
                            uint8_t (&output)[32]) {
    memset(output, 0, sizeof(output));
    if (!length) return 0;
    write<uint32_t>(output, 0, identifier);
    if (offset <= UINT32_MAX && length <= UINT32_MAX) {
        write<uint32_t>(output, 4, 1);
        write<uint32_t>(output, 8, static_cast<uint32_t>(offset));
        write<uint32_t>(output, 12, static_cast<uint32_t>(length));
        return 1;
    }
    write<uint32_t>(output, 4, 3);
    write<uint64_t>(output, 8, offset);
    write<uint32_t>(output, 16, identifier);
    write<uint32_t>(output, 20, 4);
    write<uint64_t>(output, 24, length);
    return 2;
}
}
