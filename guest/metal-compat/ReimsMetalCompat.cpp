#include <Headers/kern_api.hpp>
#include <Headers/kern_mach.hpp>
#include <Headers/kern_patcher.hpp>
#include <Headers/plugin_start.hpp>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IOSharedDataQueue.h>
#include <IOKit/pci/IOPCIDevice.h>
#include <libkern/copyio.h>
#include <sys/random.h>
#include <sys/proc.h>
#include <kern/thread_call.h>
#include "ResourceABI.hpp"
#include "SubmissionABI.hpp"
#include "DirtyABI.hpp"

class ReimsMetalClient;

namespace {
constexpr UInt32 probeClientType = 0x524d0005;
constexpr UInt32 probeVersionSelector = 0x524d0000;
using OpenClient = IOReturn (*)(IOService *, task_t, void *, UInt32,
                               OSDictionary *, IOUserClient **);
OpenClient originalOpen;
IONotifier *acceleratorNotification;
IOService *selectedAccelerator;
bool globalMetalEnabled;
OSString *modernPluginName;
uint32_t nextTraceClient, traceRecords;

bool browserGPUProcess() {
    if (!globalMetalEnabled) return false;
    char name[64] {};
    proc_selfname(name, sizeof(name));
    // Keep the desktop on its established driver; these GPU helpers need the newer Metal ABI.
    return !strcmp(name, "com.apple.WebKit.GPU") || !strcmp(name, "Google Chrome Helper") ||
        !strcmp(name, "Google Chrome Helper (GPU)");
}
using AsyncResult = IOReturn (*)(OSAsyncReference64, IOReturn, io_user_reference_t *, UInt32);
using AsyncResultOptions = IOReturn (*)(OSAsyncReference64, IOReturn, io_user_reference_t *, UInt32, IOOptionBits);
AsyncResult originalAsync;
AsyncResultOptions originalAsyncOptions;
bool callbacksRouted;
IOLock *clientsLock;
ReimsMetalClient *clients;
struct Completion {
    Completion *next;
    ReimsMetalClient *client;
    uint64_t queueCookie;
    IOReturn result;
    io_user_reference_t arguments[7];
};
Completion *firstCompletion;
Completion *lastCompletion;
thread_call_t completionCall;
using AddObject = OSObject *(*)(OSObject *, uint32_t, uint32_t, uint32_t, const void *);
AddObject originalAddObject;
const char *paravirtPaths[] = {"/System/Library/Extensions/AppleParavirtGPU.kext/Contents/MacOS/AppleParavirtGPU"};
KernelPatcher::KextInfo paravirtInfo {"com.apple.driver.AppleParavirtGPU", paravirtPaths, 1, {true}, {}, 0};
struct TextureCreation {
    TextureCreation *next;
    thread_t thread;
    const void *bytes;
    uint32_t length;
    bool consumed;
};
TextureCreation *textureCreations;

struct Notification {
    Notification *next;
    uint32_t identifier;
    IOSharedDataQueue *ring;
    IOMemoryMap *mapping;
    mach_port_t port;
};
struct Pending {
    Pending *next;
    uint32_t memory, commands;
    uint64_t completed, batch;
};
struct Queue {
    Queue *next;
    uint32_t identifier;
    uint64_t cookie;
    IOUserClient *backend;
    Notification *notification;
    Pending *pending;
};

struct MappedRange {
    IOMemoryDescriptor *descriptor;
    IOMemoryMap *mapping;
    IOReturn error;
    bool open(task_t task, uint64_t address, uint64_t length, bool writable = false) {
        error = kIOReturnBadArgument;
        if (!address || !length || address > UINT64_MAX - length) return false;
        descriptor = IOMemoryDescriptor::withAddressRange(address, length,
            kIODirectionOut | kIOMemoryHostOnly | kIOMemoryMapperNone, task);
        error = kIOReturnNoMemory;
        if (!descriptor) return false;
        // Existing device-pager mappings cannot be DMA-wired again; this is a CPU alias.
        mapping = descriptor->map(writable ? 0 : kIOMapReadOnly);
        error = mapping ? kIOReturnSuccess : kIOReturnVMError;
        return mapping != nullptr;
    }
    uint8_t *bytes() const {
        return mapping ? reinterpret_cast<uint8_t *>(mapping->getVirtualAddress()) : nullptr;
    }
    void close() {
        if (mapping) mapping->release();
        if (descriptor) descriptor->release();
        *this = {};
    }
};

struct Resource {
    Resource *next;
    uint32_t identifier;
    uint32_t objectIdentifier;
    uint64_t length;
    MappedRange kernelShared;
    IOBufferMemoryDescriptor *mirror;
    IOMemoryMap *userMapping;
    void dispose() {
        if (userMapping) userMapping->release();
        if (mirror) mirror->release();
        kernelShared.close();
        IODelete(this, Resource, 1);
    }
};

struct SharedMemory {
    SharedMemory *next;
    uint32_t identifier;
    uint32_t length;
    MappedRange range;
    void dispose() {
        range.close();
        IODelete(this, SharedMemory, 1);
    }
};

IOReturn call(IOUserClient *client, uint32_t selector,
              IOExternalMethodArguments *arguments) {
    if (!client || !arguments) return kIOReturnBadArgument;
    IOExternalMethodArguments forwarded = *arguments;
    forwarded.selector = selector;
    IOReturn status = client->externalMethod(selector, &forwarded);
    arguments->scalarOutputCount = forwarded.scalarOutputCount;
    arguments->structureOutputSize = forwarded.structureOutputSize;
    return status;
}
}

class EXPORT ReimsMetalPluginName : public OSString {
    OSDeclareDefaultStructors(ReimsMetalPluginName)
public:
    bool serialize(OSSerialize *serializer) const override {
        if (modernPluginName && browserGPUProcess()) return modernPluginName->serialize(serializer);
        return OSString::serialize(serializer);
    }
};
OSDefineMetaClassAndStructors(ReimsMetalPluginName, OSString)

class EXPORT ReimsMetalClient : public IOUserClient {
    OSDeclareDefaultStructors(ReimsMetalClient)
    IOUserClient *legacyDevice {nullptr};
    IOUserClient *legacyShared {nullptr};
    task_t owner {TASK_NULL};
    IOLock *stateLock {nullptr};
    Resource *resources {nullptr};
    SharedMemory *sharedMemory {nullptr};
    uint32_t privateBytes {0};
    bool closing {false};
    uint32_t traceClient {0};
    bool traceThisClient {false};
    uint32_t resourceStage {0};
    IOReturn resourceError {kIOReturnSuccess};
    IOExternalTrap releaseTrap {};
    IOExternalTrap submitTrap {};
    IOExternalTrap childReleaseTrap {}, dirtySetupTrap {}, dirtyFlushTrap {};
    IOBufferMemoryDescriptor *dirtyMemory {nullptr};
    IOMemoryMap *dirtyUserMapping {nullptr};
    MappedRange legacyDirtyHead {}, legacyDirtyTail {}, legacyDirtyEntries {};
    uint32_t dirtyHead {0}, legacyDirtyCapacity {0};
    static constexpr uint32_t dirtyCapacity = 256;

    void trace(const char *operation, uint32_t selector, IOReturn status,
               uint32_t size, uint32_t first, uint32_t second, uint32_t third) {
        if (!traceThisClient) return;
        uint32_t sequence = __atomic_fetch_add(&traceRecords, 1U, __ATOMIC_RELAXED);
        if (sequence >= 2048) return;
        char key[48], value[192];
        snprintf(key, sizeof(key), "ReimsMetalTrace-%04u", sequence);
        snprintf(value, sizeof(value), "client=%u %s=%u status=%x size=%u values=%u,%u,%u",
            traceClient, operation, selector, static_cast<uint32_t>(status), size, first, second, third);
        selectedAccelerator->setProperty(key, value);
    }
    ReimsMetalClient *nextClient {nullptr};
    uint64_t cookie {0};
    bool registered {false};
    uint32_t nextIdentifier {1};
    uint64_t nextBatch {1};
    Queue *queues {nullptr};
    Notification *notifications {nullptr};

    Queue *queueFor(uint64_t identifier) {
        for (auto queue = queues; queue; queue = queue->next)
            if (queue->identifier == identifier) return queue;
        return nullptr;
    }

    IOReturn releaseShared(uint64_t identifier) {
        IOExternalMethodArguments args {};
        args.scalarInput = &identifier;
        args.scalarInputCount = 1;
        return call(legacyShared, 8, &args);
    }

    void disposePending(Pending *pending) {
        if (pending->memory) releaseShared(pending->memory);
        if (pending->commands) releaseShared(pending->commands);
        IODelete(pending, Pending, 1);
    }

    IOReturn copyShared(const void *bytes, size_t size, uint32_t &identifier) {
        uint8_t reply[16] {};
        uint64_t length = (size + PAGE_SIZE - 1) & ~(static_cast<uint64_t>(PAGE_SIZE) - 1);
        IOExternalMethodArguments args {};
        args.scalarInput = &length;
        args.scalarInputCount = 1;
        args.structureOutput = reply;
        args.structureOutputSize = sizeof(reply);
        IOReturn status = call(legacyShared, 7, &args);
        if (status != kIOReturnSuccess) return status;
        identifier = ReimsABI::read<uint32_t>(reply, 12);
        MappedRange target {};
        if (args.structureOutputSize != sizeof(reply) || ReimsABI::read<uint32_t>(reply, 8) < size ||
            !target.open(owner, ReimsABI::read<uint64_t>(reply, 0), size, true)) status = kIOReturnVMError;
        else memcpy(target.bytes(), bytes, size);
        target.close();
        return status;
    }

    void disposeNotification(Notification *notification) {
        if (notification->mapping) notification->mapping->release();
        if (notification->ring) {
            notification->ring->setNotificationPort(MACH_PORT_NULL);
            notification->ring->release();
        }
        if (notification->port) releaseNotificationPort(notification->port);
        IODelete(notification, Notification, 1);
    }

    // Called without stateLock: closing a legacy queue may deliver its last callback.
    void disposeQueue(Queue *queue) {
        if (queue->backend) { queue->backend->clientClose(); queue->backend->release(); }
        while (queue->pending) {
            auto pending = queue->pending;
            queue->pending = pending->next;
            disposePending(pending);
        }
        IODelete(queue, Queue, 1);
    }

    IOReturn createQueue(IOExternalMethodArguments *args) {
        if (!callbacksRouted) return kIOReturnNotReady;
        if (args->scalarInputCount || args->scalarOutputCount ||
            args->structureInputSize != 1032 || !args->structureInput ||
            args->structureOutputSize != 16 || !args->structureOutput ||
            ReimsABI::read<uint32_t>(args->structureInput, 1028) || !nextIdentifier)
            return kIOReturnBadArgument;
        auto queue = IONewZero(Queue, 1);
        if (!queue) return kIOReturnNoMemory;
        queue->identifier = nextIdentifier++;
        do { read_random(&queue->cookie, sizeof(queue->cookie)); } while (!queue->cookie);
        IOReturn status = originalOpen(getProvider(), owner, owner, 8, nullptr, &queue->backend);
        if (status == kIOReturnSuccess) status = queue->backend->connectClient(legacyShared);
        if (status == kIOReturnSuccess) {
            OSAsyncReference64 reference {};
            setAsyncReference64(reference, MACH_PORT_NULL, cookie, queue->cookie, owner);
            IOExternalMethodArguments registration {};
            registration.asyncReference = reference;
            registration.asyncReferenceCount = 3;
            status = call(queue->backend, 0, &registration);
        }
        uint64_t trace = 0;
        if (status == kIOReturnSuccess) {
            IOExternalMethodArguments creation {};
            creation.structureInput = args->structureInput;
            creation.structureInputSize = 1028;
            creation.structureOutput = &trace;
            creation.structureOutputSize = sizeof(trace);
            status = call(queue->backend, 5, &creation);
            if (status == kIOReturnSuccess && creation.structureOutputSize != sizeof(trace))
                status = kIOReturnUnderrun;
        }
        if (status != kIOReturnSuccess) {
            // The queue has not been published and cannot receive submissions.
            if (queue->backend) { queue->backend->clientClose(); queue->backend->release(); }
            IODelete(queue, Queue, 1);
            return status;
        }
        queue->next = queues;
        queues = queue;
        ReimsABI::write<uint64_t>(args->structureOutput, 0, queue->identifier);
        ReimsABI::write<uint64_t>(args->structureOutput, 8, trace);
        return kIOReturnSuccess;
    }

    IOReturn createNotification(IOExternalMethodArguments *args) {
        if (args->scalarInputCount != 2 || !args->scalarInput || args->scalarInput[0] != 256 ||
            args->scalarInput[1] != 40 || args->structureInputSize || args->scalarOutputCount ||
            args->structureOutputSize != 16 || !args->structureOutput || !nextIdentifier)
            return kIOReturnBadArgument;
        auto notification = IONewZero(Notification, 1);
        if (!notification) return kIOReturnNoMemory;
        notification->identifier = nextIdentifier++;
        notification->ring = IOSharedDataQueue::withEntries(257, 40);
        IOMemoryDescriptor *memory = notification->ring ? notification->ring->getMemoryDescriptor() : nullptr;
        if (memory) {
            notification->mapping = memory->createMappingInTask(owner, 0, kIOMapAnywhere);
            memory->release();
        }
        if (!notification->mapping) { disposeNotification(notification); return kIOReturnNoMemory; }
        notification->next = notifications;
        notifications = notification;
        ReimsABI::write<uint64_t>(args->structureOutput, 0, notification->mapping->getAddress());
        ReimsABI::write<uint64_t>(args->structureOutput, 8, notification->identifier);
        return kIOReturnSuccess;
    }

    IOReturn createMemorylessTexture(IOExternalMethodArguments *args) {
        if (!originalAddObject) return kIOReturnNotReady;
        uint32_t command = ReimsABI::read<uint32_t>(args->structureInput, 4);
        if ((command != 1 && command != 52) ||
            ReimsABI::read<uint32_t>(args->structureInput, 8) != args->structureInputSize - 4)
            return kIOReturnBadArgument;
        TextureCreation creation {nullptr, current_thread(),
            static_cast<const uint8_t *>(args->structureInput) + 4, args->structureInputSize - 4, false};
        IOLockLock(clientsLock);
        creation.next = textureCreations;
        textureCreations = &creation;
        IOLockUnlock(clientsLock);
        IOReturn status = call(legacyShared, 256, args);
        IOLockLock(clientsLock);
        TextureCreation **link = &textureCreations;
        while (*link && *link != &creation) link = &(*link)->next;
        if (*link) *link = creation.next;
        IOLockUnlock(clientsLock);
        return status;
    }

    IOReturn initializeDirtyRing() {
        if (dirtyUserMapping) return kIOReturnSuccess;
        legacyDirtyHead.close(); legacyDirtyTail.close(); legacyDirtyEntries.close();
        if (dirtyMemory) { dirtyMemory->release(); dirtyMemory = nullptr; }
        uint8_t reply[24] {};
        IOExternalMethodArguments args {};
        args.structureOutput = reply;
        args.structureOutputSize = sizeof(reply);
        IOReturn status = call(legacyShared, 10, &args);
        if (status != kIOReturnSuccess) return status;
        if (args.structureOutputSize != sizeof(reply) ||
            !legacyDirtyHead.open(owner, ReimsABI::read<uint64_t>(reply, 0), 8) ||
            !legacyDirtyTail.open(owner, ReimsABI::read<uint64_t>(reply, 8), 4, true))
            return kIOReturnVMError;
        legacyDirtyCapacity = ReimsABI::read<uint32_t>(legacyDirtyHead.bytes(), 4);
        if (!legacyDirtyCapacity || legacyDirtyCapacity > 65536 ||
            !legacyDirtyEntries.open(owner, ReimsABI::read<uint64_t>(reply, 16),
                static_cast<uint64_t>(legacyDirtyCapacity) * 16, true)) return kIOReturnUnsupported;
        dirtyMemory = IOBufferMemoryDescriptor::inTaskWithOptions(kernel_task,
            kIODirectionInOut | kIOMemoryKernelUserShared, 2 * PAGE_SIZE, PAGE_SIZE);
        if (!dirtyMemory) return kIOReturnNoMemory;
        auto bytes = static_cast<uint8_t *>(dirtyMemory->getBytesNoCopy());
        bzero(bytes, 2 * PAGE_SIZE);
        ReimsABI::write<uint32_t>(bytes, 4, dirtyCapacity);
        dirtyUserMapping = dirtyMemory->createMappingInTask(owner, 0, kIOMapAnywhere);
        return dirtyUserMapping ? kIOReturnSuccess : kIOReturnNoMemory;
    }

    IOReturn dirtySetup(void *first, void *second, void *third, void *, void *, void *) {
        if (current_task() != owner) return kIOReturnNotPermitted;
        IOLockLock(stateLock);
        IOReturn status = closing ? kIOReturnNotOpen : initializeDirtyRing();
        if (status == kIOReturnSuccess) {
            uint64_t pointers[3] = {dirtyUserMapping->getAddress(), dirtyUserMapping->getAddress() + 8,
                dirtyUserMapping->getAddress() + 16};
            if (copyout(pointers, reinterpret_cast<user_addr_t>(first), 8) ||
                copyout(pointers + 1, reinterpret_cast<user_addr_t>(second), 8) ||
                copyout(pointers + 2, reinterpret_cast<user_addr_t>(third), 8)) status = kIOReturnBadArgument;
        }
        IOLockUnlock(stateLock);
        return status;
    }

    IOReturn flushLegacyDirtyRing() {
        IOExternalMethodArguments args {};
        return call(legacyShared, 11, &args);
    }

    IOReturn flushDirty() {
        if (!dirtyMemory) return kIOReturnSuccess;
        auto bytes = static_cast<uint8_t *>(dirtyMemory->getBytesNoCopy());
        uint32_t tail = __atomic_load_n(reinterpret_cast<uint32_t *>(bytes + 8), __ATOMIC_ACQUIRE);
        if (tail - dirtyHead > dirtyCapacity || ReimsABI::read<uint32_t>(bytes, 4) != dirtyCapacity)
            return kIOReturnBadArgument;
        while (dirtyHead != tail) {
            uint8_t entry[24], translated[32] {};
            memcpy(entry, bytes + 16 + (dirtyHead % dirtyCapacity) * 24, sizeof(entry));
            uint32_t identifier = ReimsABI::read<uint32_t>(entry, 0);
            uint32_t operation = ReimsABI::read<uint32_t>(entry, 4);
            uint64_t arg0 = ReimsABI::read<uint64_t>(entry, 8), arg1 = ReimsABI::read<uint64_t>(entry, 16);
            auto resource = resources;
            while (resource && resource->identifier != identifier) resource = resource->next;
            if (!resource) {
                trace("dirty", operation, kIOReturnNotFound, 24, identifier, static_cast<uint32_t>(arg0), static_cast<uint32_t>(arg1));
                return kIOReturnNotFound;
            }
            unsigned count = 0;
            if (operation == 0) {
                if (arg0 > resource->length || arg1 > resource->length - arg0) return kIOReturnBadArgument;
                count = ReimsABI::dirtyBuffer(identifier, arg0, arg1, translated);
            } else if (operation == 1) {
                if (!ReimsABI::dirtyTexture(identifier, arg0, arg1, translated)) return kIOReturnBadArgument;
                count = 1;
            } else if (operation == 2 && !arg0 && !arg1 && privateBytes >= 4) {
                __atomic_fetch_or(reinterpret_cast<uint32_t *>(resource->kernelShared.bytes() + 32), 1U, __ATOMIC_SEQ_CST);
            } else {
                trace("dirty", operation, kIOReturnUnsupported, 24, identifier, static_cast<uint32_t>(arg0), static_cast<uint32_t>(arg1));
                return kIOReturnUnsupported;
            }
            if (count) {
                auto headPtr = reinterpret_cast<uint32_t *>(legacyDirtyHead.bytes());
                auto tailPtr = reinterpret_cast<uint32_t *>(legacyDirtyTail.bytes());
                uint32_t head = __atomic_load_n(headPtr, __ATOMIC_ACQUIRE);
                uint32_t oldTail = __atomic_load_n(tailPtr, __ATOMIC_ACQUIRE);
                if (oldTail - head > legacyDirtyCapacity) return kIOReturnBadArgument;
                if (legacyDirtyCapacity - (oldTail - head) < count) {
                    IOReturn status = flushLegacyDirtyRing();
                    if (status != kIOReturnSuccess) return status;
                    head = __atomic_load_n(headPtr, __ATOMIC_ACQUIRE);
                    if (oldTail - head > legacyDirtyCapacity || legacyDirtyCapacity - (oldTail - head) < count)
                        return kIOReturnNoSpace;
                }
                for (unsigned i = 0; i < count; i++)
                    memcpy(legacyDirtyEntries.bytes() + ((oldTail + i) % legacyDirtyCapacity) * 16,
                        translated + i * 16, 16);
                __atomic_store_n(tailPtr, oldTail + count, __ATOMIC_RELEASE);
            }
            dirtyHead++;
            __atomic_store_n(reinterpret_cast<uint32_t *>(bytes), dirtyHead, __ATOMIC_RELEASE);
        }
        return flushLegacyDirtyRing();
    }

    IOReturn dirtyFlush(void *, void *, void *, void *, void *, void *) {
        IOLockLock(stateLock);
        IOReturn status = closing ? kIOReturnNotOpen : flushDirty();
        IOLockUnlock(stateLock);
        return status;
    }

    IOReturn releaseChild(void *parent, void *child, void *, void *, void *, void *) {
        uintptr_t parentID = reinterpret_cast<uintptr_t>(parent), childID = reinterpret_cast<uintptr_t>(child);
        if (parentID > UINT32_MAX || childID > UINT32_MAX) return kIOReturnBadArgument;
        uint32_t input[2] {static_cast<uint32_t>(parentID), static_cast<uint32_t>(childID)};
        IOExternalMethodArguments args {};
        args.structureInput = input;
        args.structureInputSize = sizeof(input);
        IOLockLock(stateLock);
        IOReturn status = closing ? kIOReturnNotOpen : flushDirty();
        if (status == kIOReturnSuccess) status = call(legacyShared, 260, &args);
        trace("child-release", 0x100, status, 8, input[0], input[1], 0);
        IOLockUnlock(stateLock);
        return status;
    }

    IOReturn releaseLegacyResource(uint64_t identifier) {
        IOExternalMethodArguments args {};
        args.scalarInput = &identifier;
        args.scalarInputCount = 1;
        return call(legacyShared, 1, &args);
    }

    IOReturn createResource(IOExternalMethodArguments *args) {
        resourceStage = 1;
        uint8_t reply[88] {};
        if (args->scalarInputCount || args->scalarOutputCount ||
            args->structureOutputSize != sizeof(reply) || !args->structureOutput ||
            !args->structureInput || args->structureInputSize > UINT32_MAX - 16)
            return kIOReturnBadArgument;
        size_t inputSize = static_cast<size_t>(args->structureInputSize) + 16;
        auto input = static_cast<uint8_t *>(IOMalloc(inputSize));
        if (!input) return kIOReturnNoMemory;
        if (!ReimsABI::resource(args->structureInput, args->structureInputSize, input, inputSize)) {
            IOFree(input, inputSize);
            return kIOReturnBadArgument;
        }
        auto resource = IONewZero(Resource, 1);
        if (!resource) { IOFree(input, inputSize); return kIOReturnNoMemory; }
        IOExternalMethodArguments forwarded {};
        forwarded.structureInput = input;
        forwarded.structureInputSize = args->structureInputSize + 16;
        forwarded.structureOutput = reply;
        forwarded.structureOutputSize = sizeof(reply);
        IOReturn status = call(legacyShared, 0, &forwarded);
        IOFree(input, inputSize);
        resourceError = status;
        MappedRange clientShared {};
        if (status == kIOReturnSuccess) {
            resourceStage = 2;
            status = kIOReturnNoMemory;
            if (forwarded.structureOutputSize == sizeof(reply) &&
                clientShared.open(owner, ReimsABI::read<uint64_t>(reply, 0x10), 260)) {
                resourceStage = 3;
                resource->identifier = ReimsABI::read<uint32_t>(clientShared.bytes(), 0x100);
                resource->length = ReimsABI::read<uint64_t>(reply, 0x28);
                resource->mirror = IOBufferMemoryDescriptor::inTaskWithOptions(kernel_task,
                    kIODirectionInOut | kIOMemoryKernelUserShared, PAGE_SIZE, PAGE_SIZE);
                if (resource->mirror && resource->kernelShared.open(owner,
                    ReimsABI::read<uint64_t>(reply, 0x18), 32 + privateBytes, true)) {
                    resourceStage = 4;
                    auto mirror = static_cast<uint8_t *>(resource->mirror->getBytesNoCopy());
                    // User mappings expose complete pages; clear every byte before publishing.
                    bzero(mirror, PAGE_SIZE);
                    memcpy(mirror, clientShared.bytes(), 128);
                    memcpy(mirror + 128, resource->kernelShared.bytes() + 32, privateBytes);
                    resource->userMapping = resource->mirror->createMappingInTask(owner, 0, kIOMapAnywhere);
                    if (resource->userMapping) status = kIOReturnSuccess;
                }
            }
        }
        if (resourceStage == 2) resourceError = clientShared.error;
        if (resourceStage == 3) resourceError = resource->kernelShared.error;
        if (resourceStage == 4) resourceError = status;
        clientShared.close();
        if (status != kIOReturnSuccess) {
            if (resource->identifier) releaseLegacyResource(resource->identifier);
            resource->dispose();
            return status;
        }
        ReimsABI::write<uint32_t>(reply, 0x24, resource->identifier);
        ReimsABI::write<uint64_t>(reply, 0x10, resource->userMapping->getAddress());
        resource->next = resources;
        resource->objectIdentifier = ReimsABI::read<uint32_t>(args->structureInput, 0x5c);
        resources = resource;
        memcpy(args->structureOutput, reply, sizeof(reply));
        resourceStage = 5;
        return kIOReturnSuccess;
    }

    IOReturn sharedAllocation(IOExternalMethodArguments *args) {
        if (args->scalarInputCount != 2 || !args->scalarInput || args->scalarInput[1] > 1 ||
            args->structureInputSize || args->scalarOutputCount ||
            args->structureOutputSize != 16 || !args->structureOutput) return kIOReturnBadArgument;
        auto memory = IONewZero(SharedMemory, 1);
        if (!memory) return kIOReturnNoMemory;
        uint8_t reply[16] {};
        IOExternalMethodArguments forwarded = *args;
        forwarded.scalarInputCount = 1;
        forwarded.structureOutput = reply;
        IOReturn status = call(legacyShared, 7, &forwarded);
        if (status == kIOReturnSuccess) {
            memory->identifier = ReimsABI::read<uint32_t>(reply, 12);
            memory->length = ReimsABI::read<uint32_t>(reply, 8);
            if (forwarded.structureOutputSize != sizeof(reply) ||
                !memory->range.open(owner, ReimsABI::read<uint64_t>(reply, 0), memory->length)) {
                uint64_t identifier = memory->identifier;
                IOExternalMethodArguments cleanup {};
                cleanup.scalarInput = &identifier;
                cleanup.scalarInputCount = 1;
                call(legacyShared, 8, &cleanup);
                status = kIOReturnNoMemory;
            }
        }
        if (status != kIOReturnSuccess) { memory->dispose(); return status; }
        memory->next = sharedMemory;
        sharedMemory = memory;
        memcpy(args->structureOutput, reply, sizeof(reply));
        return kIOReturnSuccess;
    }

    IOReturn releaseResource(void *identifier, void *, void *, void *, void *, void *) {
        IOLockLock(stateLock);
        uint64_t value = reinterpret_cast<uintptr_t>(identifier);
        Resource **link = &resources;
        while (*link && (*link)->identifier != value) link = &(*link)->next;
        IOReturn status = kIOReturnNotFound;
        if (!closing && *link) {
            uint32_t references = ReimsABI::read<uint32_t>((*link)->kernelShared.bytes(), 24);
            status = flushDirty();
            if (status == kIOReturnSuccess) status = releaseLegacyResource(value);
            trace("resource-release", 1, status, 0, (*link)->identifier, (*link)->objectIdentifier, references);
            if (status == kIOReturnSuccess) {
                Resource *resource = *link;
                *link = resource->next;
                resource->dispose();
            }
        }
        IOLockUnlock(stateLock);
        return status;
    }

    IOReturn submit(void *queueArgument, void *sizeArgument, void *entryArgument,
                    void *resultArgument, void *, void *) {
        if (reinterpret_cast<uintptr_t>(sizeArgument) != 48 || current_task() != owner)
            return kIOReturnBadArgument;
        uint8_t entry[48] {};
        if (copyin(reinterpret_cast<user_addr_t>(entryArgument), entry, sizeof(entry)))
            return kIOReturnBadArgument;
        IOLockLock(stateLock);
        IOReturn status = submitLocked(reinterpret_cast<uintptr_t>(queueArgument), entry, 1);
        trace("submit", 0, status, sizeof(entry), static_cast<uint32_t>(reinterpret_cast<uintptr_t>(queueArgument)),
            ReimsABI::read<uint32_t>(entry, 0), ReimsABI::read<uint32_t>(entry, 4));
        IOLockUnlock(stateLock);
        if (status == kIOReturnSuccess && resultArgument) {
            uint32_t value = 0;
            if (copyout(&value, reinterpret_cast<user_addr_t>(resultArgument), sizeof(value)))
                return kIOReturnBadArgument;
        }
        return status;
    }

    IOReturn prepareEntry(const uint8_t *entry, Pending *&prepared) {
        prepared = nullptr;
        uint32_t commandsID = ReimsABI::read<uint32_t>(entry, 0);
        uint32_t resourcesID = ReimsABI::read<uint32_t>(entry, 4);
        SharedMemory *commands = nullptr, *metadata = nullptr;
        for (auto memory = sharedMemory; memory; memory = memory->next) {
            if (memory->identifier == commandsID) commands = memory;
            if (memory->identifier == resourcesID) metadata = memory;
        }
        if (!commands || !metadata || metadata->length < 0x48 || commands->length < 0xc0)
            return kIOReturnBadArgument;
        // Snapshot user-writable metadata before checking lengths and resolving resource IDs.
        auto source = static_cast<uint8_t *>(IOMalloc(metadata->length));
        if (!source) return kIOReturnNoMemory;
        memcpy(source, metadata->range.bytes(), metadata->length);
        auto commandSnapshot = static_cast<uint8_t *>(IOMalloc(commands->length));
        if (!commandSnapshot) { IOFree(source, metadata->length); return kIOReturnNoMemory; }
        memcpy(commandSnapshot, commands->range.bytes(), commands->length);
        size_t size = ReimsABI::submissionSize(source, metadata->length, commandSnapshot, commands->length);
        auto translated = size ? static_cast<uint8_t *>(IOMalloc(size)) : nullptr;
        bool valid = translated && ReimsABI::submission(source, metadata->length,
            commandSnapshot, commands->length, translated, size, this,
            [](void *context, uint32_t identifier, uint64_t &length) {
                auto client = static_cast<ReimsMetalClient *>(context);
                for (auto resource = client->resources;
                     resource; resource = resource->next)
                    if (resource->identifier == identifier) {
                        length = resource->length;
                        return true;
                    }
                client->trace("lookup-missing", 0, kIOReturnUnsupported, 0, identifier,
                    static_cast<uint32_t>(length), 0);
                return false;
            }, commandSnapshot);
        if (!valid) {
            trace("metadata-rejected", size != 0, kIOReturnUnsupported, metadata->length,
                ReimsABI::read<uint32_t>(source, 0x20), ReimsABI::read<uint32_t>(source, 0x24),
                ReimsABI::read<uint32_t>(source, 0x0c));
            trace("first-record", 0, 0, ReimsABI::read<uint32_t>(source, 0x40),
                ReimsABI::read<uint32_t>(source, 0x30), ReimsABI::read<uint32_t>(source, 0x34),
                ReimsABI::read<uint32_t>(source, 0x44));
            uint32_t diagnosticCount = ReimsABI::read<uint32_t>(source, 0x40);
            for (uint32_t i = 0; i < diagnosticCount && i < 16; i++) {
                size_t offset = 0x48 + (i / 6) * 64;
                if (offset > metadata->length || metadata->length - offset < 64) break;
                const uint8_t *chunk = source + offset;
                uint32_t id = ReimsABI::read<uint32_t>(chunk, (i % 6) * 4);
                trace("entry", i, 0, ReimsABI::read<uint16_t>(chunk, 48 + (i % 6) * 2), id,
                    ReimsABI::read<uint32_t>(chunk, 24 + (i % 6) * 4), ReimsABI::read<uint32_t>(chunk, 60));
                for (auto resource = resources; resource; resource = resource->next)
                    if (resource->identifier == id)
                        trace("allocation", i, 0, 0, id, static_cast<uint32_t>(resource->length),
                            static_cast<uint32_t>(resource->length >> 32));
            }
            size_t next = 0x18 + (ReimsABI::read<uint32_t>(source, 0x24) & 0x7fffffffU);
            if (next <= metadata->length && metadata->length - next >= 24) {
                uint32_t begin = ReimsABI::read<uint32_t>(source, next + 16);
                trace("next-header", 0, 0, ReimsABI::read<uint32_t>(source, next + 8),
                    ReimsABI::read<uint32_t>(source, next + 12), begin,
                    ReimsABI::read<uint32_t>(source, next + 20));
                if (begin <= commands->length && commands->length - begin >= 8)
                    trace("next-command", 0, 0, begin, ReimsABI::read<uint32_t>(commandSnapshot, begin),
                        ReimsABI::read<uint32_t>(commandSnapshot, begin + 4), 0);
            }
            for (size_t offset = 8; offset < 0xac; offset++) {
                if (commandSnapshot[offset]) {
                    trace("preamble-field", 0, 0, static_cast<uint32_t>(offset), commandSnapshot[offset], 0, 0);
                    break;
                }
            }
        }
        IOFree(source, metadata->length);
        if (!valid) {
            IOFree(commandSnapshot, commands->length);
            if (translated) IOFree(translated, size);
            return kIOReturnUnsupported;
        }
        auto pending = IONewZero(Pending, 1);
        if (!pending) {
            IOFree(translated, size);
            IOFree(commandSnapshot, commands->length);
            return kIOReturnNoMemory;
        }
        IOReturn status = copyShared(translated, size, pending->memory);
        if (status == kIOReturnSuccess)
            status = copyShared(commandSnapshot, commands->length, pending->commands);
        IOFree(translated, size);
        IOFree(commandSnapshot, commands->length);
        if (status != kIOReturnSuccess) {
            disposePending(pending);
            return status;
        }
        pending->completed = ReimsABI::read<uint64_t>(entry, 24);
        prepared = pending;
        return kIOReturnSuccess;
    }

    IOReturn submitLocked(uint64_t queueID, const uint8_t *entries, uint32_t count) {
        auto queue = queueFor(queueID);
        if (closing || !queue || !queue->notification || !nextBatch) return kIOReturnNotReady;
        size_t size = ReimsABI::batchSize(count, 48, static_cast<uint64_t>(count) * 48);
        if (!size || !entries) return kIOReturnBadArgument;
        IOReturn status = flushDirty();
        if (status != kIOReturnSuccess) return status;
        auto submission = static_cast<uint8_t *>(IOMalloc(size));
        if (!submission) return kIOReturnNoMemory;
        bzero(submission, size);
        ReimsABI::write<uint32_t>(submission, 4, count);
        uint64_t batch = nextBatch++;
        Pending *prepared = nullptr;
        for (uint32_t i = 0; i < count; i++) {
            const uint8_t *entry = entries + static_cast<size_t>(i) * 48;
            Pending *pending = nullptr;
            status = prepareEntry(entry, pending);
            if (status != kIOReturnSuccess) break;
            pending->batch = batch;
            pending->next = prepared;
            prepared = pending;
            uint8_t *destination = submission + 8 + static_cast<size_t>(i) * 24;
            ReimsABI::write<uint32_t>(destination, 0, pending->commands);
            ReimsABI::write<uint32_t>(destination, 4, pending->memory);
            memcpy(destination + 8, entry + 16, 16);
        }
        if (status != kIOReturnSuccess) {
            while (prepared) {
                auto pending = prepared;
                prepared = pending->next;
                disposePending(pending);
            }
            IOFree(submission, size);
            return status;
        }
        // Prepare every entry before the single legacy batch call can submit any GPU work.
        while (prepared) {
            auto pending = prepared;
            prepared = pending->next;
            pending->next = queue->pending;
            queue->pending = pending;
        }
        IOUserClient *backend = queue->backend;
        backend->retain();
        IOExternalMethodArguments args {};
        args.structureInput = submission;
        args.structureInputSize = static_cast<uint32_t>(size);
        IOLockUnlock(stateLock);
        status = call(backend, 1, &args);
        backend->release();
        IOFree(submission, size);
        IOLockLock(stateLock);
        queue = queueFor(queueID);
        if (status != kIOReturnSuccess && queue) {
            Pending **link = &queue->pending;
            while (*link) {
                auto pending = *link;
                if (pending->batch == batch) {
                    *link = pending->next;
                    disposePending(pending);
                } else link = &pending->next;
            }
        }
        return status;
    }

public:
    static mach_vm_address_t asyncAddress() {
        return reinterpret_cast<mach_vm_address_t>(&IOUserClient::sendAsyncResult64);
    }
    static mach_vm_address_t asyncOptionsAddress() {
        return reinterpret_cast<mach_vm_address_t>(&IOUserClient::sendAsyncResult64WithOptions);
    }

    static bool deliver(OSAsyncReference64 reference, IOReturn result,
                        io_user_reference_t *arguments, UInt32 count, IOReturn &status) {
        if (!clientsLock || !reference) return false;
        IOLockLock(clientsLock);
        auto client = clients;
        while (client && client->cookie != reference[1]) client = client->nextClient;
        if (client) client->retain();
        IOLockUnlock(clientsLock);
        if (!client) return false;
        if (!arguments || count != 7) {
            client->release();
            status = kIOReturnBadArgument;
            return true;
        }
        auto completion = IONewZero(Completion, 1);
        if (!completion) { client->release(); status = kIOReturnNoMemory; return true; }
        completion->client = client;
        completion->queueCookie = reference[2];
        completion->result = result;
        memcpy(completion->arguments, arguments, sizeof(completion->arguments));
        IOLockLock(clientsLock);
        if (lastCompletion) lastCompletion->next = completion;
        else firstCompletion = completion;
        lastCompletion = completion;
        IOLockUnlock(clientsLock);
        // Legacy callbacks run under driver locks; releasing memory there inverts lock order.
        thread_call_enter(completionCall);
        status = kIOReturnSuccess;
        return true;
    }

    static void drainCompletions(thread_call_param_t, thread_call_param_t) {
        for (;;) {
            IOLockLock(clientsLock);
            auto completion = firstCompletion;
            if (completion) {
                firstCompletion = completion->next;
                if (!firstCompletion) lastCompletion = nullptr;
            }
            IOLockUnlock(clientsLock);
            if (!completion) return;
            completion->client->complete(completion);
            completion->client->release();
            IODelete(completion, Completion, 1);
        }
    }

    void complete(const Completion *completion) {
        auto arguments = completion->arguments;
        IOLockLock(stateLock);
        auto queue = queues;
        while (queue && queue->cookie != completion->queueCookie) queue = queue->next;
        if (!closing && queue && queue->notification) {
            for (auto resource = resources; resource; resource = resource->next)
                memcpy(static_cast<uint8_t *>(resource->mirror->getBytesNoCopy()) + 128,
                    resource->kernelShared.bytes() + 32, privateBytes);
            Pending **link = &queue->pending;
            while (*link && (*link)->completed != arguments[0]) link = &(*link)->next;
            trace("completion", 0, completion->result, 40,
                queue->identifier, static_cast<uint32_t>(arguments[6]), *link != nullptr);
            if (*link) {
                auto pending = *link;
                *link = pending->next;
                disposePending(pending);
            }
            uint32_t result = completion->result ? static_cast<uint32_t>(completion->result)
                : static_cast<uint32_t>(arguments[6]);
            uint64_t message[5] = {arguments[0], arguments[2], arguments[4], result, 0};
            if (!queue->notification->ring->enqueue(message, sizeof(message)))
                SYSLOG("queue", "completion ring is full");
        }
        IOLockUnlock(stateLock);
    }

    IOReturn openBackends(IOService *provider, task_t task, void *security,
                         OSDictionary *properties) {
        owner = task;
        if (checkKernelArgument("-reimsmetaldbg"))
            traceClient = __atomic_add_fetch(&nextTraceClient, 1U, __ATOMIC_RELAXED);
        traceThisClient = traceClient != 0;
        stateLock = IOLockAlloc();
        if (!stateLock) return kIOReturnNoMemory;
        releaseTrap.object = this;
        releaseTrap.func = reinterpret_cast<IOTrap>(&ReimsMetalClient::releaseResource);
        submitTrap.object = this;
        submitTrap.func = reinterpret_cast<IOTrap>(&ReimsMetalClient::submit);
        childReleaseTrap.object = dirtySetupTrap.object = dirtyFlushTrap.object = this;
        childReleaseTrap.func = reinterpret_cast<IOTrap>(&ReimsMetalClient::releaseChild);
        dirtySetupTrap.func = reinterpret_cast<IOTrap>(&ReimsMetalClient::dirtySetup);
        dirtyFlushTrap.func = reinterpret_cast<IOTrap>(&ReimsMetalClient::dirtyFlush);
        IOReturn status = originalOpen(provider, task, security, 5, properties,
                                      &legacyDevice);
        if (status != kIOReturnSuccess) return status;
        status = originalOpen(provider, task, security, 6, properties, &legacyShared);
        if (status != kIOReturnSuccess) return status;
        uint8_t info[600] {};
        IOExternalMethodArguments args {};
        args.structureOutput = info;
        args.structureOutputSize = sizeof(info);
        status = call(legacyDevice, 2, &args);
        if (status != kIOReturnSuccess) return status;
        privateBytes = ReimsABI::read<uint32_t>(info, 12);
        if (args.structureOutputSize != sizeof(info) || privateBytes > 64) return kIOReturnUnsupported;
        if (!clientsLock) return kIOReturnNotReady;
        do { read_random(&cookie, sizeof(cookie)); } while (!cookie);
        IOLockLock(clientsLock);
        nextClient = clients;
        clients = this;
        registered = true;
        IOLockUnlock(clientsLock);
        return kIOReturnSuccess;
    }

    IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *args,
                            IOExternalMethodDispatch *, OSObject *, void *) override {
        IOLockLock(stateLock);
        IOReturn status = closing ? kIOReturnNotOpen : dispatch(selector, args);
        if (args && (selector == 9 || selector >= 256) && args->structureInput && args->structureInputSize >= 8) {
            auto input = args->structureInput;
            uint32_t first = ReimsABI::read<uint32_t>(input, 0), second = ReimsABI::read<uint32_t>(input, 4);
            uint32_t third = args->structureInputSize >= 12 ? ReimsABI::read<uint32_t>(input, 8) : 0;
            if (selector == 9 && args->structureInputSize >= 96) {
                first = ReimsABI::read<uint32_t>(input, 0x5c);
                second = ReimsABI::read<uint32_t>(input, 0x58);
                third = status == kIOReturnSuccess ? ReimsABI::read<uint32_t>(args->structureOutput, 0x24) : 0;
            }
            trace("method", selector, status, args->structureInputSize, first, second, third);
        }
        if (status != kIOReturnSuccess && selector < 512 && args && selectedAccelerator) {
            char key[48], value[160];
            snprintf(key, sizeof(key), "ReimsMetalFailure-%u", selector);
            uint32_t field00 = 0, field30 = 0, field58 = 0;
            if (selector == 9 && args->structureInput && args->structureInputSize >= 96) {
                field00 = ReimsABI::read<uint32_t>(args->structureInput, 0);
                field30 = ReimsABI::read<uint32_t>(args->structureInput, 0x30);
                field58 = ReimsABI::read<uint32_t>(args->structureInput, 0x58);
            }
            snprintf(value, sizeof(value), "in=%u out=%u status=%x field_00=%x field_30=%x field_58=%x",
                args->structureInputSize, args->structureOutputSize, static_cast<uint32_t>(status), field00, field30, field58);
            selectedAccelerator->setProperty(key, value);
        }
        IOLockUnlock(stateLock);
        return status;
    }

    IOReturn dispatch(uint32_t selector, IOExternalMethodArguments *args) {
        if (!args || args->structureInputDescriptor || args->structureOutputDescriptor ||
            args->asyncReferenceCount) return kIOReturnUnsupported;
        if (selector == probeVersionSelector && !args->scalarInputCount &&
            !args->structureInputSize && args->scalarOutputCount == 4 && args->scalarOutput) {
            args->scalarOutput[0] = 0x524d434f;
            args->scalarOutput[1] = 3;
            args->scalarOutput[2] = legacyDevice != nullptr;
            args->scalarOutput[3] = legacyShared != nullptr;
            return kIOReturnSuccess;
        }
        if (selector == probeVersionSelector + 1 && !args->scalarInputCount &&
            !args->structureInputSize && args->scalarOutputCount == 4 && args->scalarOutput) {
            args->scalarOutput[0] = resourceStage;
            args->scalarOutput[1] = static_cast<uint32_t>(resourceError);
            args->scalarOutput[2] = privateBytes;
            args->scalarOutput[3] = 0;
            return kIOReturnSuccess;
        }
        if (selector == 0 && !args->scalarInputCount && !args->structureInputSize &&
            args->structureOutput && args->structureOutputSize == 64)
            return call(legacyDevice, 0, args);
        if (selector == 2 && !args->scalarInputCount && !args->structureInputSize &&
            args->structureOutput && args->structureOutputSize == 536) {
            uint8_t legacy[600] {};
            IOExternalMethodArguments forwarded = *args;
            forwarded.structureOutput = legacy;
            forwarded.structureOutputSize = sizeof(legacy);
            IOReturn status = call(legacyDevice, 2, &forwarded);
            if (status != kIOReturnSuccess) return status;
            if (forwarded.structureOutputSize != sizeof(legacy)) return kIOReturnUnderrun;
            memcpy(args->structureOutput, legacy, 24);
            memcpy(static_cast<uint8_t *>(args->structureOutput) + 24, legacy + 88, 512);
            ReimsABI::write<uint32_t>(args->structureOutput, 8, privateBytes);
            return kIOReturnSuccess;
        }
        if (selector == 265 && !args->scalarInputCount && !args->structureInputSize &&
            args->structureOutput && args->structureOutputSize == 284)
            return call(legacyShared, 261, args);
        if (selector == 9) return createResource(args);
        if (selector == 14) return sharedAllocation(args);
        if (selector == 7) return createQueue(args);
        if (selector == 16) return createNotification(args);
        if (selector == 29) {
            if (args->scalarInputCount != 4 || !args->scalarInput ||
                args->scalarOutputCount != 1 || !args->scalarOutput ||
                !args->structureInput || args->structureOutputSize ||
                !ReimsABI::batchSize(args->scalarInput[2], args->scalarInput[3], args->structureInputSize))
                return kIOReturnBadArgument;
            if (args->scalarInput[1]) return kIOReturnUnsupported;
            IOReturn status = submitLocked(args->scalarInput[0],
                static_cast<const uint8_t *>(args->structureInput), static_cast<uint32_t>(args->scalarInput[2]));
            trace("batch", 29, status, static_cast<uint32_t>(args->structureInputSize),
                static_cast<uint32_t>(args->scalarInput[0]), static_cast<uint32_t>(args->scalarInput[2]), 48);
            if (status == kIOReturnSuccess) args->scalarOutput[0] = 0;
            return status;
        }
        if (selector == 28 && args->scalarInputCount == 2 && args->scalarInput &&
            !args->structureInputSize && !args->scalarOutputCount && !args->structureOutputSize) {
            auto queue = queueFor(args->scalarInput[0]);
            auto notification = notifications;
            while (notification && notification->identifier != args->scalarInput[1]) notification = notification->next;
            if (!queue || !notification) return kIOReturnNotFound;
            if (queue->pending) return kIOReturnBusy;
            queue->notification = notification;
            return kIOReturnSuccess;
        }
        if ((selector == 8 || selector == 17) && args->scalarInputCount == 1 && args->scalarInput &&
            !args->structureInputSize && !args->scalarOutputCount && !args->structureOutputSize) {
            if (selector == 8) {
                Queue **link = &queues;
                while (*link && (*link)->identifier != args->scalarInput[0]) link = &(*link)->next;
                if (!*link) return kIOReturnNotFound;
                auto queue = *link;
                if (queue->pending) return kIOReturnBusy;
                *link = queue->next;
                IOLockUnlock(stateLock);
                disposeQueue(queue);
                IOLockLock(stateLock);
            } else {
                Notification **link = &notifications;
                while (*link && (*link)->identifier != args->scalarInput[0]) link = &(*link)->next;
                if (!*link) return kIOReturnNotFound;
                for (auto queue = queues; queue; queue = queue->next)
                    if (queue->notification == *link && queue->pending) return kIOReturnBusy;
                for (auto queue = queues; queue; queue = queue->next)
                    if (queue->notification == *link) queue->notification = nullptr;
                auto notification = *link;
                *link = notification->next;
                disposeNotification(notification);
            }
            return kIOReturnSuccess;
        }
        if (selector == 15 && args->scalarInputCount == 1 && args->scalarInput &&
            !args->structureInputSize && !args->scalarOutputCount && !args->structureOutputSize) {
            SharedMemory **link = &sharedMemory;
            while (*link && (*link)->identifier != args->scalarInput[0]) link = &(*link)->next;
            if (!*link) return kIOReturnNotFound;
            IOReturn status = call(legacyShared, 8, args);
            if (status == kIOReturnSuccess) {
                SharedMemory *memory = *link;
                *link = memory->next;
                memory->dispose();
            }
            return status;
        }
        if (!args->scalarInputCount && !args->structureOutputSize && args->structureInput) {
            size_t size = args->structureInputSize;
            if (args->scalarOutputCount == 1 && args->scalarOutput) {
                if (selector == 256 && size >= 16) return createMemorylessTexture(args);
                // The legacy dispatcher validates the length-bearing serializer for all state kinds.
                if (selector == 257 && size >= 16)
                    return call(legacyShared, 256, args);
                if (selector == 260 && size == 44) return call(legacyShared, 258, args);
                if (selector == 266 && size == 56) return call(legacyShared, 262, args);
            }
            if (!args->scalarOutputCount) {
                if (selector == 258 && size == 8) {
                    uint32_t deletion[5] {ReimsABI::read<uint32_t>(args->structureInput, 0),
                        ReimsABI::read<uint32_t>(args->structureInput, 4), 1001, 12,
                        ReimsABI::read<uint32_t>(args->structureInput, 0)};
                    IOExternalMethodArguments forwarded = *args;
                    forwarded.structureInput = deletion;
                    forwarded.structureInputSize = sizeof(deletion);
                    return call(legacyShared, 257, &forwarded);
                }
                if (selector == 259 && size == 20) return call(legacyShared, 257, args);
                if (selector == 261 && size >= 16) return call(legacyShared, 259, args);
                if (selector == 262 && size == 8) return call(legacyShared, 260, args);
            }
        }
        return kIOReturnUnsupported;
    }

    IOExternalTrap *getTargetAndTrapForIndex(IOService **target, UInt32 index) override {
        if (!target || (index > 1 && index != 0x100 && index != 0x101 && index != 0x102)) return nullptr;
        *target = this;
        if (index == 0x100) return &childReleaseTrap;
        if (index == 0x101) return &dirtySetupTrap;
        if (index == 0x102) return &dirtyFlushTrap;
        return index == 0 ? &submitTrap : &releaseTrap;
    }

    IOReturn registerNotificationPort(mach_port_t port, UInt32 type, io_user_reference_t reference) override {
        IOLockLock(stateLock);
        auto notification = notifications;
        while (notification && notification->identifier != reference) notification = notification->next;
        IOReturn status = kIOReturnBadArgument;
        if (!closing && !type && notification) {
            notification->ring->setNotificationPort(port);
            if (notification->port) releaseNotificationPort(notification->port);
            notification->port = port;
            status = kIOReturnSuccess;
        } else if (port) releaseNotificationPort(port);
        IOLockUnlock(stateLock);
        return status;
    }

    void unregister() {
        if (!registered) return;
        IOLockLock(clientsLock);
        ReimsMetalClient **link = &clients;
        while (*link && *link != this) link = &(*link)->nextClient;
        if (*link) *link = nextClient;
        registered = false;
        IOLockUnlock(clientsLock);
    }

    IOReturn clientClose() override {
        IOLockLock(stateLock);
        closing = true;
        IOLockUnlock(stateLock);
        unregister();
        terminate();
        return kIOReturnSuccess;
    }

    void free() override {
        unregister();
        while (queues) { auto queue = queues; queues = queue->next; disposeQueue(queue); }
        while (notifications) { auto notification = notifications; notifications = notification->next; disposeNotification(notification); }
        if (dirtyUserMapping) dirtyUserMapping->release();
        if (dirtyMemory) dirtyMemory->release();
        legacyDirtyHead.close(); legacyDirtyTail.close(); legacyDirtyEntries.close();
        while (resources) { auto resource = resources; resources = resource->next; resource->dispose(); }
        while (sharedMemory) { auto memory = sharedMemory; sharedMemory = memory->next; memory->dispose(); }
        if (legacyShared) { legacyShared->clientClose(); legacyShared->release(); }
        if (legacyDevice) { legacyDevice->clientClose(); legacyDevice->release(); }
        if (stateLock) IOLockFree(stateLock);
        IOUserClient::free();
    }
};
OSDefineMetaClassAndStructors(ReimsMetalClient, IOUserClient)

namespace {
OSObject *addObject(OSObject *heap, uint32_t type, uint32_t identifier,
                    uint32_t length, const void *bytes) {
    bool refused = false;
    if (type == 7 && clientsLock) {
        IOLockLock(clientsLock);
        auto creation = textureCreations;
        while (creation && creation->thread != current_thread()) creation = creation->next;
        if (creation) {
            if (!creation->consumed && creation->bytes == bytes && creation->length == length) {
                // Host object type 8 is a serialized texture; type 7 is a Metal state object.
                type = 8;
                creation->consumed = true;
            } else refused = true;
        }
        IOLockUnlock(clientsLock);
    }
    return refused ? nullptr : originalAddObject(heap, type, identifier, length, bytes);
}

void paravirtLoaded(void *, KernelPatcher &patcher, size_t identifier,
                    mach_vm_address_t address, size_t size) {
    if (identifier != paravirtInfo.loadIndex || originalAddObject) return;
    KernelPatcher::RouteRequest route {
        "__ZN25AppleParavirtResourceHeap9addObjectE13APVObjectTypejjPKv", addObject, originalAddObject};
    if (!patcher.routeMultiple(identifier, &route, 1, address, size)) {
        SYSLOG("init", "serialized-texture routing unavailable");
        patcher.clearError();
    }
}

IOReturn asyncResult(OSAsyncReference64 reference, IOReturn result,
                     io_user_reference_t *arguments, UInt32 count) {
    IOReturn status;
    if (ReimsMetalClient::deliver(reference, result, arguments, count, status)) return status;
    return originalAsync(reference, result, arguments, count);
}
IOReturn asyncResultOptions(OSAsyncReference64 reference, IOReturn result,
                            io_user_reference_t *arguments, UInt32 count, IOOptionBits options) {
    IOReturn status;
    if (ReimsMetalClient::deliver(reference, result, arguments, count, status)) return status;
    return originalAsyncOptions(reference, result, arguments, count, options);
}
IOReturn openClient(IOService *provider, task_t task, void *security, UInt32 type,
                     OSDictionary *properties, IOUserClient **handler) {
    if (provider != selectedAccelerator ||
        (type != probeClientType && !(browserGPUProcess() && type == 5)))
        return originalOpen(provider, task, security, type, properties, handler);
    if (!handler) return kIOReturnBadArgument;
    *handler = nullptr;
    auto client = new ReimsMetalClient;
    if (!client) return kIOReturnNoMemory;
    if (!client->initWithTask(task, security, type, properties)) {
        client->release();
        return kIOReturnNoResources;
    }
    client->setProperty("IOUserClientDefaultLocking", true);
    if (!client->attach(provider)) { client->release(); return kIOReturnNoResources; }
    IOReturn status = client->openBackends(provider, task, security, properties);
    if (status != kIOReturnSuccess || !client->start(provider)) {
        client->detach(provider);
        client->release();
        return status != kIOReturnSuccess ? status : kIOReturnNoResources;
    }
    *handler = client;
    SYSLOG("client", "diagnostic connection opened");
    return kIOReturnSuccess;
}

bool matched(void *, void *, IOService *service, IONotifier *) {
    if (originalOpen) return true;
    bool selected = false;
    for (IOService *node = service; node; node = node->getProvider()) {
        auto pci = OSDynamicCast(IOPCIDevice, node);
        if (pci) {
            selected = pci->configRead16(kIOPCIConfigVendorID) == 0x106b &&
                       pci->configRead16(kIOPCIConfigDeviceID) == 0xeeee;
            break;
        }
    }
    if (!selected) return true;
    selectedAccelerator = service;
    selectedAccelerator->retain();
    using Member = IOReturn (IOService::*)(task_t, void *, UInt32,
                                          OSDictionary *, IOUserClient **);
    Member member = static_cast<Member>(&IOService::newUserClient);
    struct { uintptr_t entry; intptr_t adjustment; } representation {};
    static_assert(sizeof(member) == sizeof(representation), "unsupported member-pointer ABI");
    memcpy(&representation, &member, sizeof(member));
    if (!(representation.entry & 1) || representation.adjustment ||
        (representation.entry - 1) % sizeof(uintptr_t)) {
        SYSLOG("init", "unsupported virtual-method representation");
        return true;
    }
    size_t index = (representation.entry - 1) / sizeof(uintptr_t);
    if (MachInfo::setKernelWriting(true, KernelPatcher::kernelWriteLock) != KERN_SUCCESS) {
        SYSLOG("init", "could not enable protected vtable update");
        return true;
    }
    bool installed = KernelPatcher::routeVirtual(service, index, openClient, &originalOpen);
    MachInfo::setKernelWriting(false, KernelPatcher::kernelWriteLock);
    SYSLOG("init", "diagnostic client interceptor %s", installed ? "installed" : "unavailable");
    if (installed && callbacksRouted && checkKernelArgument("-reimsmetal")) {
        auto name = new ReimsMetalPluginName;
        modernPluginName = OSString::withCString("AppleParavirtGPUMetalIOGPUFamily");
        if (name && modernPluginName && name->initWithCString("AppleParavirtGPUMetal")) {
            globalMetalEnabled = true;
            service->setProperty("MetalPluginName", name);
        }
        if (name) name->release();
    }
    return true;
}

void patcherReady(void *, KernelPatcher &patcher) {
    clientsLock = IOLockAlloc();
    if (!clientsLock) return;
    // A re-entered thread call can otherwise deliver completed before scheduled.
    completionCall = thread_call_allocate_with_options(ReimsMetalClient::drainCompletions,
        nullptr, THREAD_CALL_PRIORITY_KERNEL, THREAD_CALL_OPTIONS_ONCE);
    if (!completionCall) return;
    KernelPatcher::RouteRequest routes[] = {
        {nullptr, asyncResult, originalAsync},
        {nullptr, asyncResultOptions, originalAsyncOptions}
    };
    routes[0].from = ReimsMetalClient::asyncAddress();
    routes[1].from = ReimsMetalClient::asyncOptionsAddress();
    callbacksRouted = patcher.routeMultiple(KernelPatcher::KernelID, routes);
    if (!callbacksRouted) { SYSLOG("init", "completion routing unavailable"); patcher.clearError(); }
    OSDictionary *matching = IOService::serviceMatching("AppleParavirtAccelerator");
    if (!matching) return;
    acceleratorNotification = IOService::addMatchingNotification(
        gIOMatchedNotification, matching, matched, nullptr);
    matching->release();
    if (!acceleratorNotification) SYSLOG("init", "accelerator notification unavailable");
}

void initialize() {
    if (lilu.onKextLoad(&paravirtInfo, 1, paravirtLoaded) != LiluAPI::Error::NoError)
        SYSLOG("init", "paravirtual driver registration failed");
    if (lilu.onPatcherLoad(patcherReady) != LiluAPI::Error::NoError)
        SYSLOG("init", "patcher registration failed");
}
const char *disableArguments[] = {"-reimsmetaloff"};
const char *debugArguments[] = {"-reimsmetaldbg"};
}

PluginConfiguration ADDPR(config) {
    xStringify(PRODUCT_NAME), parseModuleVersion(xStringify(MODULE_VERSION)),
    LiluAPI::AllowNormal, disableArguments, arrsize(disableArguments),
    debugArguments, arrsize(debugArguments), nullptr, 0,
    KernelVersion::Tahoe, KernelVersion::Tahoe, initialize
};
