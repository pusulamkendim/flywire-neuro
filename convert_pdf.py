"""Convert preprint.md to PDF via pandoc HTML + weasyprint."""
import subprocess
import os

# Step 1: Convert md to HTML with pandoc (with embedded images)
subprocess.run([
    'pandoc', 'preprint.md',
    '-o', 'preprint_temp.html',
    '--standalone',
    '--metadata', 'title=',
    '--embed-resources',
], check=True)

# Step 2: Add academic styling and convert to PDF
html_content = open('preprint_temp.html').read()

# Inject academic CSS
css = """
<style>
    body {
        font-family: 'Times New Roman', Times, Georgia, serif;
        max-width: 800px;
        margin: 40px auto;
        padding: 0 20px;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
    }
    h1 { font-size: 18pt; text-align: center; margin-bottom: 5px; }
    h2 { font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }
    h3 { font-size: 12pt; margin-top: 20px; }
    h4 { font-size: 11pt; }
    p { text-align: justify; margin: 8px 0; }
    table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 9pt; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
    th { background-color: #f5f5f5; font-weight: bold; }
    img { max-width: 100%; height: auto; margin: 15px 0; display: block; }
    hr { border: none; border-top: 1px solid #ccc; margin: 20px 0; }
    code { font-size: 9pt; }
    figcaption, img + em { font-size: 9pt; color: #666; }
    @page { size: A4; margin: 2.5cm; }
</style>
"""

html_content = html_content.replace('</head>', css + '</head>')

with open('preprint_styled.html', 'w') as f:
    f.write(html_content)

# Step 3: Convert to PDF with weasyprint
from weasyprint import HTML
HTML('preprint_styled.html').write_pdf('preprint.pdf')

# Cleanup
os.remove('preprint_temp.html')
os.remove('preprint_styled.html')

print("PDF generated: preprint.pdf")
