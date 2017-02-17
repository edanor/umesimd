#ifndef UME_SIMD_TRAITS_SPECIALIZATION_H_
#define UME_SIMD_TRAITS_SPECIALIZATION_H_

namespace UME {
namespace SIMD {

    // This class provides traits specialization required by ISATraits
    class PluginTraits {
    public:
        // The 'PluginTraits' should be provided by every plugin.
        // Using this fallbacks allows for more flexible handling of
        // plugin (or architecture) specific features. It also makes it
        // more abstract, and keeps the user-interface local in terms of
        // file hierarchy. It also keeps plugin-specific code outside the interface
        // files.
        template<typename SCALAR_TYPE>
        UME_FORCE_INLINE static constexpr unsigned int NativeLength(); // Do not define to cause compilation error on instantiation
    };

    // Specialize for given scalars
    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<uint8_t> () {
        return 1;
    }

    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<uint16_t> () {
        return 1;
    }

    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<uint32_t> () {
        return 1;
    }

    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<uint64_t> () {
        return 1;
    }
    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<int8_t> () {
        return 1;
    }

    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<int16_t> () {
        return 1;
    }

    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<int32_t> () {
        return 1;
    }

    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<int64_t> () {
        return 1;
    }

    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<float> () {
        return 1;
    }

    template<>
    UME_FORCE_INLINE constexpr unsigned int PluginTraits::NativeLength<double> () {
        return 1;
    }

}
}
#endif
