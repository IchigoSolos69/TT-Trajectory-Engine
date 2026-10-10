"""Index read-only research evidence; never change source PDFs."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
local = json.loads((root / "local_pdf_manifest.json").read_text(encoding="utf-8"))
references = [
    {"id": "A1", "title": "BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking", "year": 2025, "url": "https://arxiv.org/abs/2509.18387v1", "status": "attached_full_text", "source": local[0]["path"], "pages": 13, "visually_inspected_pages": [1, 5, 8, 11], "sha256": local[0]["sha256"]},
    {"id": "A2", "title": "Video-Based Table Tennis Tracking and Trajectory Prediction Using Convolutional Neural Networks", "year": 2022, "url": "https://doi.org/10.1142/S0218348X22401569", "status": "attached_full_text", "source": local[1]["path"], "pages": 23, "visually_inspected_pages": [2, 8, 17], "sha256": local[1]["sha256"]},
    {"id": "B1", "title": "Reliable Real-Time Ball Tracking for Robot Table Tennis", "year": 2019, "url": "https://www.mdpi.com/2218-6581/8/4/90", "status": "publisher_blocked_corresponding_author_preprint_reviewed", "full_text_url": "https://arxiv.org/pdf/1908.07332", "artifact": "robotics_arxiv.pdf"},
    {"id": "B2", "title": "Fast moving table tennis ball tracking algorithm based on graph neural network", "year": 2024, "url": "https://www.nature.com/articles/s41598-024-80056-3", "status": "publisher_pdf_reviewed", "full_text_url": "https://www.nature.com/articles/s41598-024-80056-3.pdf", "artifact": "nature_pdf.pdf"},
    {"id": "B3", "title": "Ball Tracking and Trajectory Prediction for Table-Tennis Robots", "year": 2020, "url": "https://www.mdpi.com/1424-8220/20/2/333", "status": "publisher_pdf_reviewed", "full_text_url": "https://mdpi-res.com/d_attachment/sensors/sensors-20-00333/article_deploy/sensors-20-00333.pdf", "artifact": "sensors_publisher.pdf", "visually_inspected_pages": [14, 16]},
    {"id": "B4", "title": "Table tennis players use superior saccadic eye movements to track moving visual targets", "year": 2024, "url": "https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1289800/full", "status": "publisher_full_text_reviewed"},
    {"id": "B5", "title": None, "year": None, "url": "https://ieeexplore.ieee.org/abstract/document/11147615", "status": "publisher_blocked_identity_unresolved", "reason": "Abstract, document, stamp and metadata routes blocked or unavailable; exact-ID search provided no verified match. No method inferred."},
    {"id": "B6", "title": None, "year": None, "url": "https://ieeexplore.ieee.org/abstract/document/10572328", "status": "publisher_blocked_identity_unresolved", "reason": "Publisher verification/blocking and rate-limited metadata lookup; exact-ID search did not resolve identity. Related event-camera reference not equated with this ID."},
    {"id": "B7", "title": "Research on real–time tracking of table tennis ball based on machine learning with low-speed camera", "year": 2018, "url": "https://www.tandfonline.com/doi/full/10.1080/21642583.2018.1450167#d1e239", "status": "primary_search_cache_full_text_reviewed_direct_pdf_blocked"},
    {"id": "B8", "title": None, "year": None, "url": "https://ieeexplore.ieee.org/abstract/document/4281948", "status": "publisher_blocked_identity_unresolved", "reason": "Publisher access blocked; exact-ID searches had no trustworthy bibliographic match. No year or method guessed."},
    {"id": "B9", "title": "Studies and simulations on the flight trajectories of spinning table tennis ball via high-speed camera vision tracking system", "year": 2019, "online_year": 2018, "url": "https://journals.sagepub.com/doi/abs/10.1177/1754337118815021", "status": "restricted_access_abstract_only", "limitation": "Full methods, equations, hardware details and error protocol unreviewed."},
    {"id": "B10", "title": "An On-Line Algorithm for Measuring the Translational and Rotational Velocities of a Table Tennis Ball", "year": 2012, "url": "https://www.tandfonline.com/doi/abs/10.9746/jcmsi.5.233", "status": "primary_indexed_abstract_first_page_only", "limitation": "Complete publisher PDF retrieval blocked; full methods and errors unreviewed."},
    {"id": "B11", "title": "Tracking a table tennis ball for umpiring purposes", "year": 2015, "url": "https://ieeexplore.ieee.org/abstract/document/7153160", "status": "publisher_blocked_author_accepted_manuscript_reviewed", "full_text_url": "https://researchportal.port.ac.uk/files/10388418/Tracking_a_table_tennis_ball_for_umpiring_purposes.pdf"},
    {"id": "B12", "title": "Application of Table Tennis Ball Trajectory and Rotation-Oriented Prediction Algorithm Using Artificial Intelligence", "year": 2022, "url": "https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.820028/full", "status": "publisher_full_text_reviewed", "artifact": "frontiers_prediction.html"},
    {"id": "C1", "title": "Table Tennis Ball Spin Estimation with an Event Camera", "year": 2024, "url": "https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Gossard_Table_Tennis_Ball_Spin_Estimation_with_an_Event_Camera_CVPRW_2024_paper.html", "status": "additional_verified_primary_full_text", "artifact": "eventcamera_pdf.pdf", "limitation": "Requires event-camera acquisition; not identified as B5, B6 or B8."},
]

assets = []
for path in sorted(root.iterdir()):
    if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".png", ".html"}:
        assets.append({"file": path.name, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "warning": "Bot-verification response; not paper full text" if path.name == "sensors_pmc.html" else None})

manifest = {
    "review_date": "2026-10-09",
    "report": "RESEARCH_APPLICATIONS.md",
    "references": references,
    "assets": assets,
    "scope": "Methods/evidence review only. Local PDF sources unmodified. No accuracy promotion or application-code changes.",
    "acquisition_notes": [
        "Every requested specific URL was attempted with browsing; selected primary alternative versions were used when available.",
        "Python certificate-trust errors were resolved using PowerShell's normal TLS trust, not by disabling certificate verification.",
        "source_fetch_manifest.json records the final automated fetch batch only; this manifest is the complete consolidated access ledger.",
        "Research PDFs and extracted text are local review evidence; this manifest does not authorize redistribution or inclusion in application packages.",
    ],
}
(root / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Indexed {len(references)} references and {len(assets)} evidence files.")
