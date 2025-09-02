#  india-vote

**india-vote (Independent Network for Democratic Inclusion and Accountability – Voter Oversight & Transparency in Elections)** is a civic-tech project focused on building open, reliable, and accessible pipelines around India’s electoral data.

----


### ▶️ Usage

The script to download all electoral roll files for a district is located at:


Run the script with the following command:

```bash
python scripts/download/eroll_dt_all.py --s 02 --dt 21
```
Note: CAPTCHA will have to reloaded and entered after every roll type is downloaded entirely, and press ENTER in the terminal to continue downloading.

----
## 🎯 Goals
- Enable **transparent access** to Assembly Constituency (AC) and Part-wise electoral rolls.  
- Provide **structured datasets** (CSV/Parquet) for researchers, journalists, and civic groups.  
- Build **interactive dashboards & maps** (H3 grids, booth-level summaries) to track voter distribution and missing data.  
- Support **automation pipelines** for scraping, parsing (OCR), and updating rolls regularly.  

---

## 🛠️ Features
- Automated **download & parsing** of electoral rolls from the Election Commission of India.  
- **OCR & translation pipelines** (Hindi → English) for digitizing voter lists.  
- **Geospatial integration** with constituency boundaries and polling station locations.  
- **ETL workflows** for clean, use-ready datasets.  
- CLI & Streamlit apps for **AC-level tracking and visualization**.  

---

## 📦 Tech Stack
- **Python**: `requests`, `selenium-wire`, `pandas`, `geopandas`, `tesseract/TrOCR`, `layoutlmv3`  
- **ETL & Storage**: CSV/Parquet, Google BigQuery connectors  
- **Visualization**: Streamlit, H3 hexagon maps, Plotly  
- **Automation**: GitHub Actions for monthly refresh  

---

## 🌍 Impact
By making voter roll data **more transparent, structured, and accessible**, `india-vote` aims to strengthen civic participation, election monitoring, and democratic accountability in India.
