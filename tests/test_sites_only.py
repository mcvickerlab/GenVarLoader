import pytest
import sys
import os

# genvarloader directory
genvarloader_path = '/blue/juannanzhou/fahimeh.rahimi/GenVarLoader/python'
sys.path.append(genvarloader_path)

from genvarloader._variants._sitesonly import SitesOnlyVCF, apply_site_only_variants

# synthetic VCF file
VCF_FILE = "/blue/juannanzhou/fahimeh.rahimi/GenVarLoader/tests/data/vcf/synthetic_sites_only_test_data.vcf"

# Test case: Test the VCF file conversion to a BED-like DataFrame
def test_sitesonly_to_bedlike():
   
    vcf_file = SitesOnlyVCF(VCF_FILE)
    
    # Convert the VCF data to a BED-like DataFrame
    df = vcf_file.to_bedlike()

    
    print(df) 
    assert df.shape[0] > 0 


def test_sitesonly_info_field():
    vcf_file = SitesOnlyVCF(
        VCF_FILE, lambda v: v.INFO["AF"] > 0.05, attributes=["ID"], info_fields=["AF"]
    )
    
    
    df = vcf_file.to_bedlike()
    print(df)  
    assert "AF" in df.columns  


def test_apply_variants():
    haps = [] 
    v_idxs = []  
    ref_coords = []  
    site_starts = [] 
    alt_alleles = []  
    alt_offsets = []  
    desired_v_idxs = []
    desired_ref_coords = []
    desired_flags = []

  
    actual_haps, actual_v_idxs, actual_ref_coords, actual_flags = apply_site_only_variants(
        haps, v_idxs, ref_coords, site_starts, alt_alleles, alt_offsets
    )
    
    # Check if the actual results match the expected ones
    import numpy as np
    np.testing.assert_equal(actual_haps, desired_haps)
    np.testing.assert_equal(actual_v_idxs, desired_v_idxs)
    np.testing.assert_equal(actual_ref_coords, desired_ref_coords)
    np.testing.assert_equal(actual_flags, desired_flags)

if __name__ == "__main__":
    pytest.main()
