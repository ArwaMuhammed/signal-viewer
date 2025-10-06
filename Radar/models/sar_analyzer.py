import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import rasterio

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def normalize_to_percentile(data, lower=2, upper=98):
    """Normalize data to percentile range for better visualization"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return data
    p_lower = np.percentile(valid_data, lower)
    p_upper = np.percentile(valid_data, upper)
    normalized = (data - p_lower) / (p_upper - p_lower)
    return np.clip(normalized, 0, 1)

def analyze_sar_file(vv_path, vh_path=None):
    """
    Visualize SAR image from TIFF file with enhanced contrast

    Args:
        vv_path: Path to VV polarization TIFF file
        vh_path: Path to VH polarization TIFF file (optional)

    Returns:
        Dictionary containing visualizations
    """
    try:
        # Read VV polarization with diagnostics
        with rasterio.open(vv_path) as src:
            print(f"File info: {src.count} bands, dtype: {src.dtypes[0]}")
            print(f"Image size: {src.width} x {src.height}")
            vv_data = src.read(1).astype(float)
            print(f"Raw data range: {np.nanmin(vv_data)} to {np.nanmax(vv_data)}")
            print(f"Non-zero values: {np.count_nonzero(vv_data)}/{vv_data.size}")

        # Handle no-data values - but keep small values
        vv_data = np.where(vv_data <= 0, np.nan, vv_data)

        # Convert DN to backscatter (dB)
        max_val = np.nanmax(vv_data)
        print(f"Max value after cleaning: {max_val}")

        # Sentinel-1 GRD DN values need to be converted to sigma0
        # DN values are typically 0-32767 (uint16)
        # Convert to power then to dB
        # Formula: sigma0_dB = 10 * log10(DN^2) - calibration_constant
        # For visualization, we use simplified conversion

        if max_val > 100:  # DN values
            # Square the DN values to get power, then convert to dB
            vv_data = 10 * np.log10(vv_data**2 + 1)
            print(f"Converted DN to dB, new range: {np.nanmin(vv_data):.2f} to {np.nanmax(vv_data):.2f}")
        else:
            print("Data appears to be already calibrated")

        # Read VH polarization if available
        vh_data = None
        if vh_path:
            with rasterio.open(vh_path) as src:
                vh_data = src.read(1).astype(float)
            vh_data = np.where(vh_data == 0, np.nan, vh_data)
            if np.nanmax(vh_data) > 10:
                vh_data = 20 * np.log10(vh_data + 1e-10)

        # Normalize data for better visualization
        vv_normalized = normalize_to_percentile(vv_data)

        # Create main visualization
        if vh_data is not None:
            vh_normalized = normalize_to_percentile(vh_data)

            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

            # VV Polarization (normalized)
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(vv_normalized, cmap='gray', vmin=0, vmax=1)
            ax1.set_title('VV Polarization (2-98% Scaled)', fontsize=14, weight='bold', pad=10)
            ax1.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Normalized Intensity', fontsize=11)

            # VH Polarization (normalized)
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(vh_normalized, cmap='gray', vmin=0, vmax=1)
            ax2.set_title('VH Polarization (2-98% Scaled)', fontsize=14, weight='bold', pad=10)
            ax2.axis('off')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Normalized Intensity', fontsize=11)

            # Combined histogram
            ax4 = fig.add_subplot(gs[1, :])
            valid_vv = vv_data[np.isfinite(vv_data)]
            valid_vh = vh_data[np.isfinite(vh_data)]
            ax4.hist(valid_vv.flatten(), bins=100, alpha=0.6, label='VV', color='blue', edgecolor='black', linewidth=0.5)
            ax4.hist(valid_vh.flatten(), bins=100, alpha=0.6, label='VH', color='red', edgecolor='black', linewidth=0.5)
            ax4.set_xlabel('Backscatter (dB)', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.set_title('Backscatter Distribution', fontsize=14, weight='bold', pad=10)
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3, linestyle='--')

        else:
            # Single polarization - larger display
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

            # Main SAR image (normalized)
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(vv_normalized, cmap='gray', vmin=0, vmax=1)
            ax1.set_title('SAR Backscatter (2-98% Scaled)', fontsize=16, weight='bold', pad=15)
            ax1.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Normalized Intensity', fontsize=12)

            # Histogram
            ax2 = fig.add_subplot(gs[0, 1])
            valid_data = vv_data[np.isfinite(vv_data)]
            ax2.hist(valid_data.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.set_xlabel('Backscatter (dB)', fontsize=13)
            ax2.set_ylabel('Frequency', fontsize=13)
            ax2.set_title('Intensity Distribution', fontsize=16, weight='bold', pad=15)
            ax2.grid(True, alpha=0.3, linestyle='--')

            # Add statistics text
            stats_text = f'Mean: {np.nanmean(vv_data):.2f} dB'
            stats_text += f'Std: {np.nanstd(vv_data):.2f} dB'
            stats_text += f'Min: {np.nanmin(vv_data):.2f} dB'
            stats_text += f'Max: {np.nanmax(vv_data):.2f} dB'
            ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
                    fontsize=11, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(pad=2.0)
        image1 = fig_to_base64(fig)

        # Additional estimators
        # 1. Coefficient of Variation (Texture Indicator)
        cov_vv = np.nanstd(vv_data) / np.nanmean(vv_data) if np.nanmean(vv_data) != 0 else 0

        # 2. Simple Urban Index (High VV backscatter areas)
        urban_threshold = np.nanpercentile(vv_data, 90)  # Top 10% of VV backscatter
        urban_mask = vv_data > urban_threshold
        urban_fraction = np.sum(urban_mask) / np.sum(np.isfinite(vv_data)) * 100

        # Statistics
        results = {
            'statistics': {
            'Image Size': f"{vv_data.shape[0]} x {vv_data.shape[1]} pixels",
            'VV Mean': f"{np.nanmean(vv_data):.2f} dB",
            'VV Std Dev': f"{np.nanstd(vv_data):.2f} dB",
            'VV Min': f"{np.nanmin(vv_data):.2f} dB",
            'VV Max': f"{np.nanmax(vv_data):.2f} dB",
            'VV Coefficient of Variation': f"{cov_vv:.3f}",
            'Urban Area Fraction (VV > 90th percentile)': f"{urban_fraction:.1f}%"
            },
        'image1': image1,
        'image2': ''  # Empty since we removed the second visualization
        }

        if vh_data is not None:
            cov_vh = np.nanstd(vh_data) / np.nanmean(vh_data) if np.nanmean(vh_data) != 0 else 0
            results['statistics']['VH Mean'] = f"{np.nanmean(vh_data):.2f} dB"
            results['statistics']['VH Std Dev'] = f"{np.nanstd(vh_data):.2f} dB"
            results['statistics']['VH Coefficient of Variation'] = f"{cov_vh:.3f}"

        return results

    except Exception as e:
    # Return error information
        return {
        'statistics': {'Error': str(e)},
        'image1': '',
        'image2': ''
    }