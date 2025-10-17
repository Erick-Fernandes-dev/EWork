# create_spec_pdf.py
from fpdf import FPDF
import os

PDF_TEXT = [
    (1, """
    Alpha-X Pro Router - Technical Specifications

    1. Interfaces
    - WAN: 1x 2.5 Gigabit Ethernet (RJ-45)
    - LAN: 4x 1 Gigabit Ethernet (RJ-45)
    - USB: 1x USB 3.0 for external storage or modem.
    """),
    (2, """
    2. Power Consumption
    - Idle: 5W
    - Maximum Load: 25W
    - Power Adapter: 12V, 2.5A DC

    3. Operational Environment
    - Operating temperature: 0°C to 40°C
    - Storage temperature: -20°C to 60°C
    - Humidity: 10% to 90% non-condensing
    """),
    (3, """
    4. Firmware Update
    - Method: Automatic via cloud or manual upload via Web UI.
    - Procedure: Download firmware from official site, navigate to the 'System' -> 'Firmware Update' section of the Web UI, and select the file. The device will restart after the update.
    - A USB drive can also be used for recovery.
    """),
    (4, """
    5. Certifications
    - FCC, CE, RoHS
    - Wi-Fi Alliance Certified
    """)
]

def create_pdf():
    if not os.path.exists("docs"):
        os.makedirs("docs")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page_num, text in PDF_TEXT:
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, f"Page {page_num}", 0, 0, 'C')

    pdf.output("./specs.pdf")
    print("./specs.pdf created.")

if __name__ == "__main__":
    create_pdf()