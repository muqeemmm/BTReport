import numpy as np
import nibabel as nib
import attrs
import matplotlib.pyplot as plt

from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    map_coordinates,
)


@attrs.define
class NiftiImage:
    path: str
    spacing: tuple[float, float, float] = None
    canonical: bool = True
    img: nib.Nifti1Image = attrs.field(init=False)
    array: np.ndarray = attrs.field(init=False)

    def __attrs_post_init__(self):
        img = nib.load(self.path)
        if self.canonical:
            img = nib.as_closest_canonical(img)
        self.img = img
        self.array = np.asanyarray(self.img.dataobj)
        if self.spacing is None:
            self.spacing = tuple(float(x) for x in self.img.header.get_zooms()[:3])

    def update(self):
        self.img = nib.Nifti1Image(self.array, self.img.affine, self.img.header)

    def save(self, out_path=None):
        if out_path is None:
            out_path = self.path
        self.update()
        nib.save(self.img, out_path)
        return out_path


def compute_outward_normals(tc_mask, spacing):
    tc_mask = tc_mask.astype(bool)
    spacing = np.asarray(spacing, dtype=float)

    dist_in = distance_transform_edt(tc_mask, sampling=spacing)
    grads = np.gradient(dist_in, *spacing, edge_order=1)
    grad = np.stack(grads, axis=-1)

    outward = -grad
    mag = np.linalg.norm(outward, axis=-1, keepdims=True)

    outward_unit = np.divide(
        outward,
        np.maximum(mag, 1e-12),
        out=np.zeros_like(outward),
        where=mag > 1e-12,
    )
    return outward_unit


def transition_zone_thickness(
    t1ce_pth,
    seg_pth,
    step_vox=0.25,
    max_dist_mm=80.0,
):
    t1ce_img = NiftiImage(t1ce_pth, canonical=True)
    t1ce_array = t1ce_img.array.astype(np.float32)

    seg_img = NiftiImage(seg_pth, canonical=True)
    seg_array = seg_img.array.astype(np.int16)

    # canonical orientation: treat as (x, y, z)
    spacing = np.asarray(seg_img.spacing, dtype=float)

    tc_mask = np.isin(seg_array, [1, 3, 4]).astype(bool)
    ed_mask = np.isin(seg_array, [2]).astype(bool)

    # choose axial slice with largest tumor-core area
    tc_area_per_axial_slice = tc_mask.sum(axis=(0, 1))
    if np.all(tc_area_per_axial_slice == 0):
        return {
            "transition_zone_thickness_in_mm": np.nan,
            "axial_slice_index": None,
            "n_boundary_voxels": 0,
            "n_successful_rays": 0,
            "edema_intensity_band": (np.nan, np.nan),
        }

    z_best = int(np.argmax(tc_area_per_axial_slice))

    def nn_mask(mask, p):
        ix, iy, iz = np.round(p).astype(int)
        if (
            ix < 0 or iy < 0 or iz < 0
            or ix >= mask.shape[0]
            or iy >= mask.shape[1]
            or iz >= mask.shape[2]
        ):
            return False
        return bool(mask[ix, iy, iz])

    # tumor-core boundary adjacent to edema
    edema_dil = binary_dilation(ed_mask, iterations=1)
    boundary = tc_mask & edema_dil & ~ed_mask
    boundary_coords = np.argwhere(boundary)

    if boundary_coords.size == 0:
        return {
            "transition_zone_thickness_in_mm": np.nan,
            "axial_slice_index": z_best,
            "n_boundary_voxels": 0,
            "n_successful_rays": 0,
            "edema_intensity_band": (np.nan, np.nan),
        }

    # robust edema intensity band
    ed_vals = t1ce_array[ed_mask]
    if ed_vals.size == 0:
        return {
            "transition_zone_thickness_in_mm": np.nan,
            "axial_slice_index": z_best,
            "n_boundary_voxels": int(len(boundary_coords)),
            "n_successful_rays": 0,
            "edema_intensity_band": (np.nan, np.nan),
        }

    ed_med = np.median(ed_vals)
    # mad = np.median(np.abs(ed_vals - ed_med)) + 1e-6
    # abs_band = max(1.4826 * mad, 1e-6)
    # ed_lower = max(ed_med - abs_band, float(np.min(ed_vals)))
    # ed_upper = min(ed_med + abs_band, float(np.max(ed_vals)))
    p40, p60 = np.percentile(ed_vals, [40, 60])
    ed_lower = float(p40)
    ed_upper = float(p60)

    outward_normals = compute_outward_normals(tc_mask, spacing)

    thicknesses = []

    for xyz in boundary_coords:
        x0, y0, z0 = xyz.astype(float)

        u = outward_normals[tuple(xyz)].astype(float)
        nu = np.linalg.norm(u)
        if nu < 1e-8:
            continue
        u = u / nu

        mm_per_vox_step = np.linalg.norm(u * spacing)
        if mm_per_vox_step == 0:
            continue

        n_steps = int(np.ceil(max_dist_mm / (step_vox * mm_per_vox_step)))

        # true boundary point for display
        p_boundary = np.array([x0, y0, z0], dtype=float)

        # start sampling just outside tumor core
        p0 = p_boundary + u * step_vox

        prev_int = map_coordinates(
            t1ce_array,
            [[p0[0]], [p0[1]], [p0[2]]],
            order=1,
            mode="nearest",
        )[0]
        prev_d_mm = 0.0
        sampled_points = [p0.copy()]

        # accept zero only if immediately in edema and in edema band
        in_edema0 = nn_mask(ed_mask, p0)
        if in_edema0 and (ed_lower <= prev_int <= ed_upper):
            thicknesses.append(0.0)
            continue

        for k in range(1, n_steps + 1):
            p = p0 + u * (k * step_vox)
            sampled_points.append(p.copy())

            I = map_coordinates(
                t1ce_array,
                [[p[0]], [p[1]], [p[2]]],
                order=1,
                mode="nearest",
            )[0]

            d_mm = k * step_vox * mm_per_vox_step

            in_tc = nn_mask(tc_mask, p)
            in_edema = nn_mask(ed_mask, p)

            # once outside tumor core, ray is only valid while still inside edema
            if (not in_tc) and (not in_edema):
                break

            # valid stop must occur inside edema
            if in_edema and (ed_lower <= I <= ed_upper):
                p_hit = p.copy()

                if abs(I - prev_int) > 1e-6:
                    if prev_int > ed_upper and I <= ed_upper:
                        target = ed_upper
                    elif prev_int < ed_lower and I >= ed_lower:
                        target = ed_lower
                    else:
                        target = ed_med

                    alpha = (target - prev_int) / (I - prev_int + 1e-12)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    d_mm = prev_d_mm + alpha * (d_mm - prev_d_mm)

                    p_prev = sampled_points[-2]
                    p_hit = p_prev + alpha * (p - p_prev)

                thicknesses.append(float(d_mm))

                break

            prev_int, prev_d_mm = I, d_mm


    result = {
        "transition_zone_thickness_in_mm": float(np.nanmedian(thicknesses)) if len(thicknesses) else np.nan,
        "axial_slice_index": z_best,
        "n_boundary_voxels": int(len(boundary_coords)),
        "n_successful_rays": int(np.sum(np.isfinite(thicknesses))),
        "edema_intensity_band": (float(ed_lower), float(ed_upper)),
    }

    return result