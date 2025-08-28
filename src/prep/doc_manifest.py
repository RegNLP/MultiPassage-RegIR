# src/doc_manifest.py
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../MultiPassage-RegulatoryRAG
DOCS_DIR = PROJECT_ROOT / "data" / "Documents"      # keep working if Documents exist locally


# Paste your list here but point to DOCS_DIR, not plain "Documents"
DOCUMENTS = [
    {"DocumentID": 3,  "title": "CoBs", "json_file_path": str(DOCS_DIR / "COBs_updated.json")},
    {"DocumentID": 1,  "title": "AML",  "json_file_path": str(DOCS_DIR / "AML_VER09.211223_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 2,  "title": "CIB",  "json_file_path": str(DOCS_DIR / "CIB_VER04.030220_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 4,  "title": "FEES", "json_file_path": str(DOCS_DIR / "FEES_VER16.181223_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 5,  "title": "FP",   "json_file_path": str(DOCS_DIR / "FP_VER01.110319_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 6,  "title": "FUNDS","json_file_path": str(DOCS_DIR / "FUNDS_VER08.040723_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 7,  "title": "GEN",  "json_file_path": str(DOCS_DIR / "GEN_VER08.181223_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 8,  "title": "GLO",  "json_file_path": str(DOCS_DIR / "GLO_VER19.181223.json")},
    {"DocumentID": 9,  "title": "IFR",  "json_file_path": str(DOCS_DIR / "IFR_VER07.181223_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 10, "title": "MIR",  "json_file_path": str(DOCS_DIR / "MIR_VER07.181223_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 11, "title": "MKT",  "json_file_path": str(DOCS_DIR / "MKT_VER08.181223_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 13, "title": "PRU",  "json_file_path": str(DOCS_DIR / "PRU_VER13.181223_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 12, "title": "PIN",  "json_file_path": str(DOCS_DIR / "PIN.json")},
    {"DocumentID": 14, "title": "BRR Regulations", "json_file_path": str(DOCS_DIR / "BRR Regulations (December 2018)_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 15, "title": "CRS Regulations 2017", "json_file_path": str(DOCS_DIR / "CRS Regulations 2017 (Consolidated_October 2023) v6_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 16, "title": "Foreign Tax Account Compliance Regulations 2022", "json_file_path": str(DOCS_DIR / "Foreign Tax Account Compliance Regulations 2022_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 17, "title": "FSMR (Consolidated December 2023)", "json_file_path": str(DOCS_DIR / "FSMR (Consolidated_December 2023)_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 18, "title": "Guidance – Regulatory Framework for Fund Managers of Venture Capital Funds", "json_file_path": str(DOCS_DIR / "Guidance – Regulatory Framework for Fund Managers of Venture Capital Funds (VER03.181223)_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 19, "title": "Guidance - Virtual Asset Activities in ADGM", "json_file_path": str(DOCS_DIR / "Guidance - Virtual Asset Activities in ADGM (VER05.181223)_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 20, "title": "ADGM Guidance - Application of English Laws", "json_file_path": str(DOCS_DIR / "ADGM_Guidance_-_Application_of_English_Laws_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 21, "title": "API - Guidance Note", "json_file_path": str(DOCS_DIR / "API - Guidance Note_Final 14 October 2019 Eng_obligations_named_entities_defined_terms_modified.json")},
    {"DocumentID": 22, "title": "CMC", "json_file_path": str(DOCS_DIR / "CMC_VER03.270922_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 23, "title": "CONF", "json_file_path": str(DOCS_DIR / "CONF_VER03.18042019_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 25, "title": "Environmental Social and Governance Disclosures Guidance", "json_file_path": str(DOCS_DIR / "Environmental Social and Governance Disclosures Guidance_VER01.040723_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 24, "title": "Draft Guidance", "json_file_path": str(DOCS_DIR / "Draft Guidance - FSRA Guiding Principles for Virtual Assets Regulation and Supervision (IA).json")},
    {"DocumentID": 26, "title": "FinTech RegLab Guidance", "json_file_path": str(DOCS_DIR / "FinTech RegLab Guidance_VER01.31082016_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 27, "title": "GPM", "json_file_path": str(DOCS_DIR / "GPM VER03.120623_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 28, "title": "Guidance - Continuous Disclosure", "json_file_path": str(DOCS_DIR / "Guidance - Continuous Disclosure_VER01.280922_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 29, "title": "Guidance - Digital Securities Offerings and Virtual  Assets under the Financial Services and Markets Regulations", "json_file_path": str(DOCS_DIR / "Guidance - Digital Securities Offerings and Virtual  Assets under the Financial Services and Markets Regulations_240220_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 30, "title": "Guidance - Disclosure Requirements for Mining Reporting Entities", "json_file_path": str(DOCS_DIR / "Guidance - Disclosure Requirements for Mining Reporting Entities_VER01.280922_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 31, "title": "Guidance - Disclosure Requirements for Petroleum Reporting Entities", "json_file_path": str(DOCS_DIR / "Guidance - Disclosure Requirements for Petroleum Reporting Entities_VER01.280922_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 32, "title": "Guidance - Private Credit Funds", "json_file_path": str(DOCS_DIR / "Guidance - Private Credit Funds_VER01.040523_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 33, "title": "Guidance - Regulation of Digital Securities Activities in ADGM", "json_file_path": str(DOCS_DIR / "Guidance  Regulation of Digital Securities Activities in ADGM_240224_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 34, "title": "Guidance - Regulation of Spot Commodities Activities in ADGM", "json_file_path": str(DOCS_DIR / "Guidance - Regulation of Spot Commodities Activities in ADGM (VER02.181223)_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 35, "title": "Guidance - Regulatory Framework for PFP and Multilateral Trading Facilities dealing with Private Capital Markets", "json_file_path": str(DOCS_DIR / "Guidance_Regulatory Framework for PFP and Multilateral Trading Facilities dealing with Private Capital Markets (VER02.181223)_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 36, "title": "SFWG Guidance on Principles for the Effective Management of Climate-related Financial Risks", "json_file_path": str(DOCS_DIR / "SFWG_Guidance on Principles for the Effective Management of Climate-related Financial Risks_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 37, "title": "Supplementary Guidance - Authorisation of Digital Investment Management (Robo-advisory) Activities", "json_file_path": str(DOCS_DIR / "Supplementary Guidance  Authorisation of Digital Investment Management (Robo-advisory) Activities_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 38, "title": "Supplementary Guidance OTCLPs", "json_file_path": str(DOCS_DIR / "Supplementary Guidance OTCLPs (VER02.181223)_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 39, "title": "Sustainable Finance Supplementary Guidance", "json_file_path": str(DOCS_DIR / "Sustainable Finance Supplementary Guidance_VER01.040723_obligations_named_entities_defined_terms.json")},
    {"DocumentID": 40, "title": "UAE CRS Guidance Notes", "json_file_path": str(DOCS_DIR / "UAE_CRS_Guidance_Notes_17 June 2020 (002)_obligations_named_entities_defined_terms.json")},
]
