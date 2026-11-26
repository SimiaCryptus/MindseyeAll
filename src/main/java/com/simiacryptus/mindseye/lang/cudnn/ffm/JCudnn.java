package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.List;

import static java.lang.foreign.ValueLayout.*;

/**
 * Java 22 Foreign Function &amp; Memory (FFM) bindings for the NVIDIA cuDNN C API ({@code libcudnn}).
 */
@RefIgnore
public final class JCudnn {
  private static volatile boolean exceptionsEnabled = false;

  private JCudnn() {
  }

  public static void setExceptionsEnabled(boolean enabled) {
    exceptionsEnabled = enabled;
  }

  private static final class Native {
    static final SymbolLookup LIB = NativeLibraryLoader.loadLibrary("NVIDIA cuDNN (libcudnn)", List.of(
        "libcudnn.so.9",
        "/usr/local/cuda-12.6/lib64/libcudnn.so.9",
        "/usr/local/cuda/lib64/libcudnn.so.9",
        "libcudnn.so",
        "/usr/local/cuda/lib64/libcudnn.so",
        "cudnn64_9.dll",
        "cudnn"
    ));

    // Version
    static final MethodHandle cudnnGetVersion = NativeLibraryLoader.downcall(LIB, "cudnnGetVersion",
        FunctionDescriptor.of(JAVA_LONG));
    static final MethodHandle cudnnGetCudartVersion = NativeLibraryLoader.downcall(LIB, "cudnnGetCudartVersion",
        FunctionDescriptor.of(JAVA_LONG));

    // Handle lifecycle
    static final MethodHandle cudnnCreate = NativeLibraryLoader.downcall(LIB, "cudnnCreate",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnDestroy = NativeLibraryLoader.downcall(LIB, "cudnnDestroy",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));

    // Descriptor create/destroy
    static final MethodHandle cudnnCreateTensorDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnCreateTensorDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnDestroyTensorDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnDestroyTensorDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnCreateFilterDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnCreateFilterDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnDestroyFilterDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnDestroyFilterDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnCreateConvolutionDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnCreateConvolutionDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnDestroyConvolutionDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnDestroyConvolutionDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnCreatePoolingDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnCreatePoolingDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnDestroyPoolingDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnDestroyPoolingDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnCreateActivationDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnCreateActivationDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnDestroyActivationDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnDestroyActivationDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnCreateLRNDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnCreateLRNDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnDestroyLRNDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnDestroyLRNDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnCreateOpTensorDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnCreateOpTensorDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnDestroyOpTensorDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnDestroyOpTensorDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnCreateReduceTensorDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnCreateReduceTensorDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudnnDestroyReduceTensorDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnDestroyReduceTensorDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));

    // Descriptor setters
    static final MethodHandle cudnnSetTensor4dDescriptorEx = NativeLibraryLoader.downcall(LIB, "cudnnSetTensor4dDescriptorEx",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
    static final MethodHandle cudnnSetFilter4dDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnSetFilter4dDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
    static final MethodHandle cudnnSetConvolution2dDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnSetConvolution2dDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
    static final MethodHandle cudnnSetActivationDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnSetActivationDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_DOUBLE));
    static final MethodHandle cudnnSetPoolingNdDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnSetPoolingNdDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnSetLRNDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnSetLRNDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE));
    static final MethodHandle cudnnSetOpTensorDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnSetOpTensorDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT));
    static final MethodHandle cudnnSetReduceTensorDescriptor = NativeLibraryLoader.downcall(LIB, "cudnnSetReduceTensorDescriptor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));

    // Output dim & workspace queries
    static final MethodHandle cudnnGetPoolingNdForwardOutputDim = NativeLibraryLoader.downcall(LIB, "cudnnGetPoolingNdForwardOutputDim",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
    static final MethodHandle cudnnGetConvolutionNdForwardOutputDim = NativeLibraryLoader.downcall(LIB, "cudnnGetConvolutionNdForwardOutputDim",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
    static final MethodHandle cudnnGetConvolutionForwardWorkspaceSize = NativeLibraryLoader.downcall(LIB, "cudnnGetConvolutionForwardWorkspaceSize",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
    static final MethodHandle cudnnGetConvolutionBackwardDataWorkspaceSize = NativeLibraryLoader.downcall(LIB, "cudnnGetConvolutionBackwardDataWorkspaceSize",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
    static final MethodHandle cudnnGetConvolutionBackwardFilterWorkspaceSize = NativeLibraryLoader.downcall(LIB, "cudnnGetConvolutionBackwardFilterWorkspaceSize",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));

    // Convolution algorithm discovery
    static final MethodHandle cudnnFindConvolutionForwardAlgorithm = NativeLibraryLoader.downcall(LIB, "cudnnFindConvolutionForwardAlgorithm",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle cudnnFindConvolutionBackwardDataAlgorithm = NativeLibraryLoader.downcall(LIB, "cudnnFindConvolutionBackwardDataAlgorithm",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle cudnnFindConvolutionBackwardFilterAlgorithm = NativeLibraryLoader.downcall(LIB, "cudnnFindConvolutionBackwardFilterAlgorithm",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));

    // Tensor & elementwise ops
    static final MethodHandle cudnnSetTensor = NativeLibraryLoader.downcall(LIB, "cudnnSetTensor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnAddTensor = NativeLibraryLoader.downcall(LIB, "cudnnAddTensor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnTransformTensor = NativeLibraryLoader.downcall(LIB, "cudnnTransformTensor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnOpTensor = NativeLibraryLoader.downcall(LIB, "cudnnOpTensor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnReduceTensor = NativeLibraryLoader.downcall(LIB, "cudnnReduceTensor",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));

    // Activation, Softmax, Pooling, LRN
    static final MethodHandle cudnnActivationForward = NativeLibraryLoader.downcall(LIB, "cudnnActivationForward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnActivationBackward = NativeLibraryLoader.downcall(LIB, "cudnnActivationBackward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnSoftmaxForward = NativeLibraryLoader.downcall(LIB, "cudnnSoftmaxForward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnSoftmaxBackward = NativeLibraryLoader.downcall(LIB, "cudnnSoftmaxBackward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnPoolingForward = NativeLibraryLoader.downcall(LIB, "cudnnPoolingForward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnPoolingBackward = NativeLibraryLoader.downcall(LIB, "cudnnPoolingBackward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnLRNCrossChannelForward = NativeLibraryLoader.downcall(LIB, "cudnnLRNCrossChannelForward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnLRNCrossChannelBackward = NativeLibraryLoader.downcall(LIB, "cudnnLRNCrossChannelBackward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));

    // Convolution ops
    static final MethodHandle cudnnConvolutionForward = NativeLibraryLoader.downcall(LIB, "cudnnConvolutionForward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnConvolutionBiasActivationForward = NativeLibraryLoader.downcall(LIB, "cudnnConvolutionBiasActivationForward",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnConvolutionBackwardBias = NativeLibraryLoader.downcall(LIB, "cudnnConvolutionBackwardBias",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnConvolutionBackwardData = NativeLibraryLoader.downcall(LIB, "cudnnConvolutionBackwardData",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cudnnConvolutionBackwardFilter = NativeLibraryLoader.downcall(LIB, "cudnnConvolutionBackwardFilter",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));
  }

  private static int check(int status) {
    if (exceptionsEnabled && status != cudnnStatus.CUDNN_STATUS_SUCCESS) {
      throw new RuntimeException("cuDNN error: " + cudnnStatus.stringFor(status));
    }
    return status;
  }

  private static int createHandle(MethodHandle mh, NativeHandle outHandle) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment ptr = arena.allocate(ADDRESS);
      int status = (int) mh.invokeExact(ptr);
      outHandle.setHandle(ptr.get(ADDRESS, 0));
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  private static int destroyHandle(MethodHandle mh, NativeHandle handle) {
    try {
      return check((int) mh.invokeExact(NativeHandle.handleOrNull(handle)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static long cudnnGetVersion() {
    try {
      return (long) Native.cudnnGetVersion.invokeExact();
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static long cudnnGetCudartVersion() {
    try {
      return (long) Native.cudnnGetCudartVersion.invokeExact();
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnCreate(cudnnHandle handle) {
    return createHandle(Native.cudnnCreate, handle);
  }

  public static int cudnnDestroy(cudnnHandle handle) {
    return destroyHandle(Native.cudnnDestroy, handle);
  }

  public static int cudnnCreateTensorDescriptor(cudnnTensorDescriptor desc) {
    return createHandle(Native.cudnnCreateTensorDescriptor, desc);
  }

  public static int cudnnDestroyTensorDescriptor(cudnnTensorDescriptor desc) {
    return destroyHandle(Native.cudnnDestroyTensorDescriptor, desc);
  }

  public static int cudnnCreateFilterDescriptor(cudnnFilterDescriptor desc) {
    return createHandle(Native.cudnnCreateFilterDescriptor, desc);
  }

  public static int cudnnDestroyFilterDescriptor(cudnnFilterDescriptor desc) {
    return destroyHandle(Native.cudnnDestroyFilterDescriptor, desc);
  }

  public static int cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor desc) {
    return createHandle(Native.cudnnCreateConvolutionDescriptor, desc);
  }

  public static int cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor desc) {
    return destroyHandle(Native.cudnnDestroyConvolutionDescriptor, desc);
  }

  public static int cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor desc) {
    return createHandle(Native.cudnnCreatePoolingDescriptor, desc);
  }

  public static int cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor desc) {
    return destroyHandle(Native.cudnnDestroyPoolingDescriptor, desc);
  }

  public static int cudnnCreateActivationDescriptor(cudnnActivationDescriptor desc) {
    return createHandle(Native.cudnnCreateActivationDescriptor, desc);
  }

  public static int cudnnDestroyActivationDescriptor(cudnnActivationDescriptor desc) {
    return destroyHandle(Native.cudnnDestroyActivationDescriptor, desc);
  }

  public static int cudnnCreateLRNDescriptor(cudnnLRNDescriptor desc) {
    return createHandle(Native.cudnnCreateLRNDescriptor, desc);
  }

  public static int cudnnDestroyLRNDescriptor(cudnnLRNDescriptor desc) {
    return destroyHandle(Native.cudnnDestroyLRNDescriptor, desc);
  }

  public static int cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor desc) {
    return createHandle(Native.cudnnCreateOpTensorDescriptor, desc);
  }

  public static int cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor desc) {
    return destroyHandle(Native.cudnnDestroyOpTensorDescriptor, desc);
  }

  public static int cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor desc) {
    return createHandle(Native.cudnnCreateReduceTensorDescriptor, desc);
  }

  public static int cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor desc) {
    return destroyHandle(Native.cudnnDestroyReduceTensorDescriptor, desc);
  }

  public static int cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor tensorDesc, int dataType,
                                                 int n, int c, int h, int w,
                                                 int nStride, int cStride, int hStride, int wStride) {
    try {
      return check((int) Native.cudnnSetTensor4dDescriptorEx.invokeExact(
          NativeHandle.handleOrNull(tensorDesc), dataType, n, c, h, w, nStride, cStride, hStride, wStride));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSetFilter4dDescriptor(cudnnFilterDescriptor filterDesc, int dataType, int format,
                                               int k, int c, int h, int w) {
    try {
      return check((int) Native.cudnnSetFilter4dDescriptor.invokeExact(
          NativeHandle.handleOrNull(filterDesc), dataType, format, k, c, h, w));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor convDesc,
                                                    int pad_h, int pad_w, int u, int v,
                                                    int dilation_h, int dilation_w, int mode, int computeType) {
    try {
      return check((int) Native.cudnnSetConvolution2dDescriptor.invokeExact(
          NativeHandle.handleOrNull(convDesc), pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSetActivationDescriptor(cudnnActivationDescriptor activationDesc,
                                                 int mode, int reluNanOpt, double coef) {
    try {
      return check((int) Native.cudnnSetActivationDescriptor.invokeExact(
          NativeHandle.handleOrNull(activationDesc), mode, reluNanOpt, coef));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor poolingDesc, int mode, int maxpoolingNanOpt,
                                                int nbDims, int[] windowDimA, int[] paddingA, int[] strideA) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment winSeg = arena.allocateFrom(JAVA_INT, windowDimA);
      MemorySegment padSeg = arena.allocateFrom(JAVA_INT, paddingA);
      MemorySegment strSeg = arena.allocateFrom(JAVA_INT, strideA);
      return check((int) Native.cudnnSetPoolingNdDescriptor.invokeExact(
          NativeHandle.handleOrNull(poolingDesc), mode, maxpoolingNanOpt, nbDims, winSeg, padSeg, strSeg));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSetLRNDescriptor(cudnnLRNDescriptor normDesc, int lrnN,
                                          double lrnAlpha, double lrnBeta, double lrnK) {
    try {
      return check((int) Native.cudnnSetLRNDescriptor.invokeExact(
          NativeHandle.handleOrNull(normDesc), lrnN, lrnAlpha, lrnBeta, lrnK));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor opTensorDesc,
                                               int opTensorOp, int opTensorCompType, int opTensorNanOpt) {
    try {
      return check((int) Native.cudnnSetOpTensorDescriptor.invokeExact(
          NativeHandle.handleOrNull(opTensorDesc), opTensorOp, opTensorCompType, opTensorNanOpt));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor reduceTensorDesc,
                                                   int reduceTensorOp, int reduceTensorCompType,
                                                   int reduceTensorNanOpt, int reduceTensorIndices,
                                                   int reduceTensorIndicesType) {
    try {
      return check((int) Native.cudnnSetReduceTensorDescriptor.invokeExact(
          NativeHandle.handleOrNull(reduceTensorDesc), reduceTensorOp, reduceTensorCompType,
          reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnGetPoolingNdForwardOutputDim(cudnnPoolingDescriptor poolingDesc,
                                                      cudnnTensorDescriptor inputTensorDesc,
                                                      int nbDims, int[] outputTensorDimA) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outSeg = arena.allocate(JAVA_INT, nbDims);
      int status = (int) Native.cudnnGetPoolingNdForwardOutputDim.invokeExact(
          NativeHandle.handleOrNull(poolingDesc), NativeHandle.handleOrNull(inputTensorDesc), nbDims, outSeg);
      if (status == cudnnStatus.CUDNN_STATUS_SUCCESS) {
        MemorySegment.copy(outSeg, JAVA_INT, 0, outputTensorDimA, 0, nbDims);
      }
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnGetConvolutionNdForwardOutputDim(cudnnConvolutionDescriptor convDesc,
                                                          cudnnTensorDescriptor inputTensorDesc,
                                                          cudnnFilterDescriptor filterDesc,
                                                          int nbDims, int[] tensorOuputDimA) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outSeg = arena.allocate(JAVA_INT, nbDims);
      int status = (int) Native.cudnnGetConvolutionNdForwardOutputDim.invokeExact(
          NativeHandle.handleOrNull(convDesc), NativeHandle.handleOrNull(inputTensorDesc),
          NativeHandle.handleOrNull(filterDesc), nbDims, outSeg);
      if (status == cudnnStatus.CUDNN_STATUS_SUCCESS) {
        MemorySegment.copy(outSeg, JAVA_INT, 0, tensorOuputDimA, 0, nbDims);
      }
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle handle,
                                                            cudnnTensorDescriptor xDesc,
                                                            cudnnFilterDescriptor wDesc,
                                                            cudnnConvolutionDescriptor convDesc,
                                                            cudnnTensorDescriptor yDesc,
                                                            int algo, long[] sizeInBytes) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_LONG);
      int status = (int) Native.cudnnGetConvolutionForwardWorkspaceSize.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(xDesc),
          NativeHandle.handleOrNull(wDesc), NativeHandle.handleOrNull(convDesc),
          NativeHandle.handleOrNull(yDesc), algo, out);
      sizeInBytes[0] = out.get(JAVA_LONG, 0);
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle handle,
                                                                 cudnnFilterDescriptor wDesc,
                                                                 cudnnTensorDescriptor dyDesc,
                                                                 cudnnConvolutionDescriptor convDesc,
                                                                 cudnnTensorDescriptor dxDesc,
                                                                 int algo, long[] sizeInBytes) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_LONG);
      int status = (int) Native.cudnnGetConvolutionBackwardDataWorkspaceSize.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(wDesc),
          NativeHandle.handleOrNull(dyDesc), NativeHandle.handleOrNull(convDesc),
          NativeHandle.handleOrNull(dxDesc), algo, out);
      sizeInBytes[0] = out.get(JAVA_LONG, 0);
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle handle,
                                                                   cudnnTensorDescriptor xDesc,
                                                                   cudnnTensorDescriptor dyDesc,
                                                                   cudnnConvolutionDescriptor convDesc,
                                                                   cudnnFilterDescriptor dwDesc,
                                                                   int algo, long[] sizeInBytes) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_LONG);
      int status = (int) Native.cudnnGetConvolutionBackwardFilterWorkspaceSize.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(xDesc),
          NativeHandle.handleOrNull(dyDesc), NativeHandle.handleOrNull(convDesc),
          NativeHandle.handleOrNull(dwDesc), algo, out);
      sizeInBytes[0] = out.get(JAVA_LONG, 0);
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnFindConvolutionForwardAlgorithm(cudnnHandle handle,
                                                         cudnnTensorDescriptor xDesc,
                                                         cudnnFilterDescriptor wDesc,
                                                         cudnnConvolutionDescriptor convDesc,
                                                         cudnnTensorDescriptor yDesc,
                                                         int requestedAlgoCount,
                                                         int[] returnedAlgoCount,
                                                         cudnnConvolutionFwdAlgoPerf[] perfResults) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment countOut = arena.allocate(JAVA_INT);
      long stride = cudnnConvolutionFwdAlgoPerf.LAYOUT.byteSize();
      MemorySegment perfBuf = arena.allocate(cudnnConvolutionFwdAlgoPerf.LAYOUT, requestedAlgoCount);
      int status = (int) Native.cudnnFindConvolutionForwardAlgorithm.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(xDesc),
          NativeHandle.handleOrNull(wDesc), NativeHandle.handleOrNull(convDesc),
          NativeHandle.handleOrNull(yDesc), requestedAlgoCount, countOut, perfBuf);
      int count = countOut.get(JAVA_INT, 0);
      returnedAlgoCount[0] = count;
      if (status == cudnnStatus.CUDNN_STATUS_SUCCESS) {
        int n = Math.min(count, perfResults.length);
        for (int i = 0; i < n; i++) {
          if (perfResults[i] == null) perfResults[i] = new cudnnConvolutionFwdAlgoPerf();
          perfResults[i].readFrom(perfBuf, (long) i * stride);
        }
      }
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle handle,
                                                              cudnnFilterDescriptor wDesc,
                                                              cudnnTensorDescriptor dyDesc,
                                                              cudnnConvolutionDescriptor convDesc,
                                                              cudnnTensorDescriptor dxDesc,
                                                              int requestedAlgoCount,
                                                              int[] returnedAlgoCount,
                                                              cudnnConvolutionBwdDataAlgoPerf[] perfResults) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment countOut = arena.allocate(JAVA_INT);
      long stride = cudnnConvolutionBwdDataAlgoPerf.LAYOUT.byteSize();
      MemorySegment perfBuf = arena.allocate(cudnnConvolutionBwdDataAlgoPerf.LAYOUT, requestedAlgoCount);
      int status = (int) Native.cudnnFindConvolutionBackwardDataAlgorithm.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(wDesc),
          NativeHandle.handleOrNull(dyDesc), NativeHandle.handleOrNull(convDesc),
          NativeHandle.handleOrNull(dxDesc), requestedAlgoCount, countOut, perfBuf);
      int count = countOut.get(JAVA_INT, 0);
      returnedAlgoCount[0] = count;
      if (status == cudnnStatus.CUDNN_STATUS_SUCCESS) {
        int n = Math.min(count, perfResults.length);
        for (int i = 0; i < n; i++) {
          if (perfResults[i] == null) perfResults[i] = new cudnnConvolutionBwdDataAlgoPerf();
          perfResults[i].readFrom(perfBuf, (long) i * stride);
        }
      }
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle handle,
                                                                cudnnTensorDescriptor xDesc,
                                                                cudnnTensorDescriptor dyDesc,
                                                                cudnnConvolutionDescriptor convDesc,
                                                                cudnnFilterDescriptor dwDesc,
                                                                int requestedAlgoCount,
                                                                int[] returnedAlgoCount,
                                                                cudnnConvolutionBwdFilterAlgoPerf[] perfResults) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment countOut = arena.allocate(JAVA_INT);
      long stride = cudnnConvolutionBwdFilterAlgoPerf.LAYOUT.byteSize();
      MemorySegment perfBuf = arena.allocate(cudnnConvolutionBwdFilterAlgoPerf.LAYOUT, requestedAlgoCount);
      int status = (int) Native.cudnnFindConvolutionBackwardFilterAlgorithm.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(xDesc),
          NativeHandle.handleOrNull(dyDesc), NativeHandle.handleOrNull(convDesc),
          NativeHandle.handleOrNull(dwDesc), requestedAlgoCount, countOut, perfBuf);
      int count = countOut.get(JAVA_INT, 0);
      returnedAlgoCount[0] = count;
      if (status == cudnnStatus.CUDNN_STATUS_SUCCESS) {
        int n = Math.min(count, perfResults.length);
        for (int i = 0; i < n; i++) {
          if (perfResults[i] == null) perfResults[i] = new cudnnConvolutionBwdFilterAlgoPerf();
          perfResults[i].readFrom(perfBuf, (long) i * stride);
        }
      }
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSetTensor(cudnnHandle handle, cudnnTensorDescriptor yDesc, Pointer y, Pointer valuePtr) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnSetTensor.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(yDesc),
          Pointer.toNativeOrNull(y, arena), Pointer.toNativeOrNull(valuePtr, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnAddTensor(cudnnHandle handle, Pointer alpha, cudnnTensorDescriptor aDesc, Pointer A,
                                   Pointer beta, cudnnTensorDescriptor cDesc, Pointer C) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnAddTensor.invokeExact(
          NativeHandle.handleOrNull(handle), Pointer.toNativeOrNull(alpha, arena),
          NativeHandle.handleOrNull(aDesc), Pointer.toNativeOrNull(A, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(cDesc),
          Pointer.toNativeOrNull(C, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnTransformTensor(cudnnHandle handle, Pointer alpha, cudnnTensorDescriptor xDesc, Pointer x,
                                         Pointer beta, cudnnTensorDescriptor yDesc, Pointer y) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnTransformTensor.invokeExact(
          NativeHandle.handleOrNull(handle), Pointer.toNativeOrNull(alpha, arena),
          NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(yDesc),
          Pointer.toNativeOrNull(y, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnOpTensor(cudnnHandle handle, cudnnOpTensorDescriptor opTensorDesc,
                                  Pointer alpha1, cudnnTensorDescriptor aDesc, Pointer A,
                                  Pointer alpha2, cudnnTensorDescriptor bDesc, Pointer B,
                                  Pointer beta, cudnnTensorDescriptor cDesc, Pointer C) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnOpTensor.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(opTensorDesc),
          Pointer.toNativeOrNull(alpha1, arena), NativeHandle.handleOrNull(aDesc), Pointer.toNativeOrNull(A, arena),
          Pointer.toNativeOrNull(alpha2, arena), NativeHandle.handleOrNull(bDesc), Pointer.toNativeOrNull(B, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(cDesc), Pointer.toNativeOrNull(C, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnReduceTensor(cudnnHandle handle, cudnnReduceTensorDescriptor reduceTensorDesc,
                                      Pointer indices, long indicesSizeInBytes,
                                      Pointer workspace, long workspaceSizeInBytes,
                                      Pointer alpha, cudnnTensorDescriptor aDesc, Pointer A,
                                      Pointer beta, cudnnTensorDescriptor cDesc, Pointer C) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnReduceTensor.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(reduceTensorDesc),
          Pointer.toNativeOrNull(indices, arena), indicesSizeInBytes,
          Pointer.toNativeOrNull(workspace, arena), workspaceSizeInBytes,
          Pointer.toNativeOrNull(alpha, arena), NativeHandle.handleOrNull(aDesc), Pointer.toNativeOrNull(A, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(cDesc), Pointer.toNativeOrNull(C, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnActivationForward(cudnnHandle handle, cudnnActivationDescriptor activationDesc,
                                           Pointer alpha, cudnnTensorDescriptor xDesc, Pointer x,
                                           Pointer beta, cudnnTensorDescriptor yDesc, Pointer y) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnActivationForward.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(activationDesc),
          Pointer.toNativeOrNull(alpha, arena), NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnActivationBackward(cudnnHandle handle, cudnnActivationDescriptor activationDesc,
                                            Pointer alpha, cudnnTensorDescriptor yDesc, Pointer y,
                                            cudnnTensorDescriptor dyDesc, Pointer dy,
                                            cudnnTensorDescriptor xDesc, Pointer x,
                                            Pointer beta, cudnnTensorDescriptor dxDesc, Pointer dx) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnActivationBackward.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(activationDesc),
          Pointer.toNativeOrNull(alpha, arena), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena),
          NativeHandle.handleOrNull(dyDesc), Pointer.toNativeOrNull(dy, arena),
          NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(dxDesc), Pointer.toNativeOrNull(dx, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSoftmaxForward(cudnnHandle handle, int algo, int mode,
                                        Pointer alpha, cudnnTensorDescriptor xDesc, Pointer x,
                                        Pointer beta, cudnnTensorDescriptor yDesc, Pointer y) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnSoftmaxForward.invokeExact(
          NativeHandle.handleOrNull(handle), algo, mode,
          Pointer.toNativeOrNull(alpha, arena), NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnSoftmaxBackward(cudnnHandle handle, int algo, int mode,
                                         Pointer alpha, cudnnTensorDescriptor yDesc, Pointer y,
                                         cudnnTensorDescriptor dyDesc, Pointer dy,
                                         Pointer beta, cudnnTensorDescriptor dxDesc, Pointer dx) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnSoftmaxBackward.invokeExact(
          NativeHandle.handleOrNull(handle), algo, mode,
          Pointer.toNativeOrNull(alpha, arena), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena),
          NativeHandle.handleOrNull(dyDesc), Pointer.toNativeOrNull(dy, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(dxDesc), Pointer.toNativeOrNull(dx, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnPoolingForward(cudnnHandle handle, cudnnPoolingDescriptor poolingDesc,
                                        Pointer alpha, cudnnTensorDescriptor xDesc, Pointer x,
                                        Pointer beta, cudnnTensorDescriptor yDesc, Pointer y) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnPoolingForward.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(poolingDesc),
          Pointer.toNativeOrNull(alpha, arena), NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnPoolingBackward(cudnnHandle handle, cudnnPoolingDescriptor poolingDesc,
                                         Pointer alpha, cudnnTensorDescriptor yDesc, Pointer y,
                                         cudnnTensorDescriptor dyDesc, Pointer dy,
                                         cudnnTensorDescriptor xDesc, Pointer x,
                                         Pointer beta, cudnnTensorDescriptor dxDesc, Pointer dx) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnPoolingBackward.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(poolingDesc),
          Pointer.toNativeOrNull(alpha, arena), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena),
          NativeHandle.handleOrNull(dyDesc), Pointer.toNativeOrNull(dy, arena),
          NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(dxDesc), Pointer.toNativeOrNull(dx, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnLRNCrossChannelForward(cudnnHandle handle, cudnnLRNDescriptor normDesc, int lrnMode,
                                                Pointer alpha, cudnnTensorDescriptor xDesc, Pointer x,
                                                Pointer beta, cudnnTensorDescriptor yDesc, Pointer y) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnLRNCrossChannelForward.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(normDesc), lrnMode,
          Pointer.toNativeOrNull(alpha, arena), NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnLRNCrossChannelBackward(cudnnHandle handle, cudnnLRNDescriptor normDesc, int lrnMode,
                                                 Pointer alpha, cudnnTensorDescriptor yDesc, Pointer y,
                                                 cudnnTensorDescriptor dyDesc, Pointer dy,
                                                 cudnnTensorDescriptor xDesc, Pointer x,
                                                 Pointer beta, cudnnTensorDescriptor dxDesc, Pointer dx) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnLRNCrossChannelBackward.invokeExact(
          NativeHandle.handleOrNull(handle), NativeHandle.handleOrNull(normDesc), lrnMode,
          Pointer.toNativeOrNull(alpha, arena), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena),
          NativeHandle.handleOrNull(dyDesc), Pointer.toNativeOrNull(dy, arena),
          NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(dxDesc), Pointer.toNativeOrNull(dx, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnConvolutionForward(cudnnHandle handle, Pointer alpha,
                                            cudnnTensorDescriptor xDesc, Pointer x,
                                            cudnnFilterDescriptor wDesc, Pointer w,
                                            cudnnConvolutionDescriptor convDesc, int algo,
                                            Pointer workSpace, long workSpaceSizeInBytes,
                                            Pointer beta, cudnnTensorDescriptor yDesc, Pointer y) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnConvolutionForward.invokeExact(
          NativeHandle.handleOrNull(handle), Pointer.toNativeOrNull(alpha, arena),
          NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          NativeHandle.handleOrNull(wDesc), Pointer.toNativeOrNull(w, arena),
          NativeHandle.handleOrNull(convDesc), algo,
          Pointer.toNativeOrNull(workSpace, arena), workSpaceSizeInBytes,
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnConvolutionBiasActivationForward(cudnnHandle handle, Pointer alpha1,
                                                          cudnnTensorDescriptor xDesc, Pointer x,
                                                          cudnnFilterDescriptor wDesc, Pointer w,
                                                          cudnnConvolutionDescriptor convDesc, int algo,
                                                          Pointer workSpace, long workSpaceSizeInBytes,
                                                          Pointer alpha2, cudnnTensorDescriptor zDesc, Pointer z,
                                                          cudnnTensorDescriptor biasDesc, Pointer bias,
                                                          cudnnActivationDescriptor activationDesc,
                                                          cudnnTensorDescriptor yDesc, Pointer y) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnConvolutionBiasActivationForward.invokeExact(
          NativeHandle.handleOrNull(handle), Pointer.toNativeOrNull(alpha1, arena),
          NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          NativeHandle.handleOrNull(wDesc), Pointer.toNativeOrNull(w, arena),
          NativeHandle.handleOrNull(convDesc), algo,
          Pointer.toNativeOrNull(workSpace, arena), workSpaceSizeInBytes,
          Pointer.toNativeOrNull(alpha2, arena), NativeHandle.handleOrNull(zDesc), Pointer.toNativeOrNull(z, arena),
          NativeHandle.handleOrNull(biasDesc), Pointer.toNativeOrNull(bias, arena),
          NativeHandle.handleOrNull(activationDesc), NativeHandle.handleOrNull(yDesc), Pointer.toNativeOrNull(y, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnConvolutionBackwardBias(cudnnHandle handle, Pointer alpha,
                                                 cudnnTensorDescriptor dyDesc, Pointer dy,
                                                 Pointer beta, cudnnTensorDescriptor dbDesc, Pointer db) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnConvolutionBackwardBias.invokeExact(
          NativeHandle.handleOrNull(handle), Pointer.toNativeOrNull(alpha, arena),
          NativeHandle.handleOrNull(dyDesc), Pointer.toNativeOrNull(dy, arena),
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(dbDesc), Pointer.toNativeOrNull(db, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnConvolutionBackwardData(cudnnHandle handle, Pointer alpha,
                                                 cudnnFilterDescriptor wDesc, Pointer w,
                                                 cudnnTensorDescriptor dyDesc, Pointer dy,
                                                 cudnnConvolutionDescriptor convDesc, int algo,
                                                 Pointer workSpace, long workSpaceSizeInBytes,
                                                 Pointer beta, cudnnTensorDescriptor dxDesc, Pointer dx) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnConvolutionBackwardData.invokeExact(
          NativeHandle.handleOrNull(handle), Pointer.toNativeOrNull(alpha, arena),
          NativeHandle.handleOrNull(wDesc), Pointer.toNativeOrNull(w, arena),
          NativeHandle.handleOrNull(dyDesc), Pointer.toNativeOrNull(dy, arena),
          NativeHandle.handleOrNull(convDesc), algo,
          Pointer.toNativeOrNull(workSpace, arena), workSpaceSizeInBytes,
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(dxDesc), Pointer.toNativeOrNull(dx, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudnnConvolutionBackwardFilter(cudnnHandle handle, Pointer alpha,
                                                   cudnnTensorDescriptor xDesc, Pointer x,
                                                   cudnnTensorDescriptor dyDesc, Pointer dy,
                                                   cudnnConvolutionDescriptor convDesc, int algo,
                                                   Pointer workSpace, long workSpaceSizeInBytes,
                                                   Pointer beta, cudnnFilterDescriptor dwDesc, Pointer dw) {
    try (Arena arena = Arena.ofConfined()) {
      return check((int) Native.cudnnConvolutionBackwardFilter.invokeExact(
          NativeHandle.handleOrNull(handle), Pointer.toNativeOrNull(alpha, arena),
          NativeHandle.handleOrNull(xDesc), Pointer.toNativeOrNull(x, arena),
          NativeHandle.handleOrNull(dyDesc), Pointer.toNativeOrNull(dy, arena),
          NativeHandle.handleOrNull(convDesc), algo,
          Pointer.toNativeOrNull(workSpace, arena), workSpaceSizeInBytes,
          Pointer.toNativeOrNull(beta, arena), NativeHandle.handleOrNull(dwDesc), Pointer.toNativeOrNull(dw, arena)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  private static RuntimeException propagate(Throwable t) {
    if (t instanceof RuntimeException re) return re;
    if (t instanceof Error err) throw err;
    return new RuntimeException(t);
  }
}
