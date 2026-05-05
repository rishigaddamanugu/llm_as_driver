#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

from panda3d.core import loadPrcFileData


def _load_panda3d_visual_quality() -> None:
    if os.environ.get("MD_PANDA_TEX_QUALITY", "").lower() not in ("1", "true", "yes", "on"):
        return
    loadPrcFileData("", "texture-minfilter linear-mipmap-linear")
    loadPrcFileData("", "texture-anisotropic-degree 8")


def _patch_metadrive_lighting_mild() -> None:
    if os.environ.get("MD_BOOST_AMBIENT", "").lower() not in ("1", "true", "yes", "on"):
        return

    from metadrive.engine.core.light import Light
    from panda3d.core import LVector4

    _orig = Light.__init__

    def __init__(self):
        _orig(self)
        self.ambient_np.node().setColor(LVector4(0.33, 0.33, 0.34, 1))

    Light.__init__ = __init__


def _patch_simplepbr_for_macos() -> None:
    if sys.platform != "darwin":
        return

    import metadrive.engine.core.engine_core as engine_core
    import metadrive.third_party.simplepbr as simplepbr
    from direct.filter.FilterManager import FilterManager

    _real_init = simplepbr.init

    def _init_for_mac(**kwargs):
        kwargs["msaa_samples"] = 0
        kwargs["use_hardware_skinning"] = False
        if os.environ.get("MD_SOFTER_PBR_EXPOSURE", "").lower() in ("1", "true", "yes", "on"):
            kwargs.setdefault("exposure", 0.74)
        return _real_init(**kwargs)

    def _setup_tonemapping_darwin(self):
        import panda3d.core as p3d

        from metadrive.third_party.simplepbr import _load_shader_str

        if self._shader_ready:
            self.manager.cleanup()
            for caster in self.get_all_casters():
                sbuff_size = caster.get_shadow_buffer_size()
                caster.set_shadow_buffer_size((0, 0))
                caster.set_shadow_buffer_size(sbuff_size)

        def _try(manager, float_color, rgba_bits, tex_format, comp_type):
            fbprops = p3d.FrameBufferProperties()
            fbprops.float_color = float_color
            fbprops.set_rgba_bits(*rgba_bits)
            fbprops.set_depth_bits(24)
            fbprops.set_multisamples(self.msaa_samples)
            tex = p3d.Texture()
            tex.set_format(tex_format)
            tex.set_component_type(comp_type)
            quad = manager.render_scene_into(colortex=tex, fbprops=fbprops)
            return quad, tex

        quad, scene_tex = _try(
            self.manager,
            True,
            (16, 16, 16, 16),
            p3d.Texture.F_rgba16,
            p3d.Texture.T_float,
        )
        if quad is None:
            print(
                "[launch_metadrive_demo] simplePBR HDR tonemap FBO failed; retrying LDR RGBA8 (macOS).",
                flush=True,
            )
            self.manager = FilterManager(self.window, self.camera_node)
            quad, scene_tex = _try(
                self.manager,
                False,
                (8, 8, 8, 8),
                p3d.Texture.F_rgba,
                p3d.Texture.T_unsigned_byte,
            )

        self.tonemap_quad = quad
        if self.tonemap_quad is None:
            raise RuntimeError(
                "simplePBR tonemap buffer failed on macOS (tried HDR float and LDR RGBA8). "
                "Try: export METADRIVE_MAC_GL=compat (OpenGL 3.2) or software GL (see drive_main tips)."
            )

        defines = {}
        if self.use_330:
            defines["USE_330"] = ""
        post_vert_str = _load_shader_str("post.vert", defines)
        post_frag_str = _load_shader_str("tonemap.frag", defines)
        tonemap_shader = p3d.Shader.make(
            p3d.Shader.SL_GLSL,
            vertex=post_vert_str,
            fragment=post_frag_str,
        )
        self.tonemap_quad.set_shader(tonemap_shader)
        self.tonemap_quad.set_shader_input("tex", scene_tex)
        self.tonemap_quad.set_shader_input("exposure", self.exposure)

    simplepbr.init = _init_for_mac
    engine_core.init = _init_for_mac
    simplepbr.Pipeline._setup_tonemapping = _setup_tonemapping_darwin


def _patch_panda3d_for_macos() -> None:
    if sys.platform != "darwin":
        return

    loadPrcFileData("", "framebuffer-multisample 0")
    loadPrcFileData("", "multisamples 0")

    gl_mode = os.environ.get("METADRIVE_MAC_GL", "").strip().lower()

    force_gl32 = gl_mode in ("compat", "legacy", "3.2", "32", "gl32")
    if force_gl32:
        loadPrcFileData("", "gl-version 3 2")

    srgb = os.environ.get("METADRIVE_FRAMEBUFFER_SRGB", "0").lower()
    if srgb in ("1", "true", "yes", "on"):
        loadPrcFileData("", "framebuffer-srgb #t")
    else:
        loadPrcFileData("", "framebuffer-srgb #f")


def apply_metadrive_render_patches() -> None:
    _load_panda3d_visual_quality()
    _patch_panda3d_for_macos()
    _patch_simplepbr_for_macos()
    _patch_metadrive_lighting_mild()


if __name__ == "__main__":
    from drive_main import main

    main()
