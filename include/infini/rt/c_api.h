#ifndef INFINI_RT_C_API_H_
#define INFINI_RT_C_API_H_

#if defined(_WIN32)
#define INFINI_RT_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) && \
    ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define INFINI_RT_EXPORT __attribute__((visibility("default")))
#else
#define INFINI_RT_EXPORT
#endif

#ifdef __cplusplus
#define INFINI_RT_EXTERN_C extern "C"
#else
#define INFINI_RT_EXTERN_C
#endif

typedef enum {
  INFINI_RT_STATUS_SUCCESS = 0,
  INFINI_RT_STATUS_INVALID_ARGUMENT = 1,
  INFINI_RT_STATUS_UNSUPPORTED_DEVICE = 2,
  INFINI_RT_STATUS_RUNTIME_ERROR = 3,
} infiniRtStatus_t;

typedef enum {
  INFINI_RT_DEVICE_CPU = 0,
  INFINI_RT_DEVICE_NVIDIA = 1,
  INFINI_RT_DEVICE_CAMBRICON = 2,
  INFINI_RT_DEVICE_ASCEND = 3,
  INFINI_RT_DEVICE_METAX = 4,
  INFINI_RT_DEVICE_MOORE = 5,
  INFINI_RT_DEVICE_ILUVATAR = 6,
  INFINI_RT_DEVICE_KUNLUN = 7,
  INFINI_RT_DEVICE_HYGON = 8,
  INFINI_RT_DEVICE_QY = 9,
} infiniRtDeviceType_t;

typedef struct {
  infiniRtDeviceType_t type;
  int index;
} infiniRtDevice_t;

typedef enum {
  INFINI_RT_STREAM_CAPTURE_MODE_GLOBAL = 0,
  INFINI_RT_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
  INFINI_RT_STREAM_CAPTURE_MODE_RELAXED = 2,
} infiniRtStreamCaptureMode_t;

typedef void* infiniRtStream_t;
typedef void* infiniRtGraph_t;
typedef void* infiniRtGraphExec_t;

INFINI_RT_EXTERN_C INFINI_RT_EXPORT infiniRtStatus_t infiniRtStreamWrap(
    infiniRtDevice_t device, void* native_stream, infiniRtStream_t* stream);

INFINI_RT_EXTERN_C INFINI_RT_EXPORT infiniRtStatus_t
infiniRtStreamDestroy(infiniRtStream_t stream);

INFINI_RT_EXTERN_C INFINI_RT_EXPORT infiniRtStatus_t infiniRtStreamBeginCapture(
    infiniRtStream_t stream, infiniRtStreamCaptureMode_t mode);

INFINI_RT_EXTERN_C INFINI_RT_EXPORT infiniRtStatus_t
infiniRtStreamEndCapture(infiniRtStream_t stream, infiniRtGraph_t* graph);

INFINI_RT_EXTERN_C INFINI_RT_EXPORT infiniRtStatus_t
infiniRtGraphDestroy(infiniRtGraph_t graph);

INFINI_RT_EXTERN_C INFINI_RT_EXPORT infiniRtStatus_t infiniRtGraphInstantiate(
    infiniRtGraphExec_t* graph_exec, infiniRtGraph_t graph);

INFINI_RT_EXTERN_C INFINI_RT_EXPORT infiniRtStatus_t
infiniRtGraphExecDestroy(infiniRtGraphExec_t graph_exec);

INFINI_RT_EXTERN_C INFINI_RT_EXPORT infiniRtStatus_t
infiniRtGraphLaunch(infiniRtGraphExec_t graph_exec, infiniRtStream_t stream);

#endif
