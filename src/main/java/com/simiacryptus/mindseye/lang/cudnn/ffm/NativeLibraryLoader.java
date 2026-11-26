package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Loads CUDA Runtime and cuDNN shared libraries via the Java 22 Foreign Function &amp; Memory (FFM) API.
 */
@RefIgnore
public final class NativeLibraryLoader {
  private static final Linker LINKER = Linker.nativeLinker();

  private NativeLibraryLoader() {
  }

  public static SymbolLookup loadLibrary(String description, List<String> candidates) {
    Throwable lastError = null;
    for (String candidate : candidates) {
      try {
        Path path = Path.of(candidate);
        if (path.isAbsolute()) {
          if (!Files.exists(path)) {
            continue;
          }
          return SymbolLookup.libraryLookup(path, Arena.global());
        } else {
          return SymbolLookup.libraryLookup(candidate, Arena.global());
        }
      } catch (Throwable t) {
        lastError = t;
      }
    }
    throw new UnsatisfiedLinkError(
        "Failed to load native library for " + description + " (tried: " + candidates + ")"
            + (lastError != null ? ": " + lastError.getMessage() : ""));
  }

  public static MethodHandle downcall(SymbolLookup lookup, String symbolName, FunctionDescriptor descriptor) {
    Optional<MemorySegment> symbol = lookup.find(symbolName);
    if (symbol.isEmpty()) {
      throw new UnsatisfiedLinkError("Native symbol not found: " + symbolName);
    }
    return LINKER.downcallHandle(symbol.get(), descriptor);
  }

  public static Optional<MethodHandle> downcallOptional(SymbolLookup lookup, String symbolName, FunctionDescriptor descriptor) {
    return lookup.find(symbolName).map(sym -> LINKER.downcallHandle(sym, descriptor));
  }
}
