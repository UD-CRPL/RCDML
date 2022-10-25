###########################################################################################################
# Tool: get_drug_names
# Purpose: Used to get the list of drugs that pertain on a certain drug family or vice versa
###########################################################################################################
import pandas as pd
from pathlib import Path

drug_family = "RTK_VEGFRs"
data_path = "/Users/mf0082/Documents/Nemours/AML/beatAML/dataset/"

def get_drug_names(data_path, drug_family):
    # Drug list with drug family information
    drug_list = pd.read_excel(data_path + "variants_BeatAML.xlsx", sheet_name="Table S11-Drug Families")
    # Drug list with sample count
    drug_count = pd.read_excel(data_path + "variants_BeatAML.xlsx", sheet_name="Table S10-Drug Responses")
    # Gets rid of any drug that has less than 300 samples, and gets rid of any text after the name of the drug
    drug_count = drug_count[['inhibitor.1', 'Sample counts']].dropna()
    drug_count = drug_count[drug_count['Sample counts'] > 300]
    drug_count = drug_count['inhibitor.1'].apply(lambda x: x.split(" ")[0])
    drug_list["inhibitor"] = drug_list["inhibitor"].apply(lambda x: x.split(" ")[0])
    # Generate sets of each drug list:
    drug_set = set(drug_list['inhibitor'].values) & set(drug_count.values)
    in_family = drug_list[drug_list["family"] == drug_family]
    in_family = set(in_family["inhibitor"].values) & set(drug_count.values)
    not_in_family =  drug_set - in_family
    return in_family, not_in_family, drug_set

family, match, drug_list = get_drug_names(data_path, drug_family)

def rtk_type_iii_match(data_path, family):
    rtk_fam = get_drug_names(data_path, "RTK_TYPE_III")[0]
    match = set(rtk_fam) & set(family)
    return match

rtk3_match = rtk_type_iii_match(data_path, family)

print("DRUGS IN FAMILY: ", len(family))
print(family)

print("DRUGS NOT IN FAMILY: ", len(match))
print(match)

print("MATCH WITH RTK-TYPE-III: ", len(rtk3_match))
print(rtk3_match)
