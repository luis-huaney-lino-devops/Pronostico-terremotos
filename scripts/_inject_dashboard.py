"""Reemplaza el payload embebido del dashboard con reports/dashboard_data.json."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "dashboard/index.html").read_text(encoding="utf-8")
data = (ROOT / "reports/dashboard_data.json").read_text(encoding="utf-8")
pat = re.compile(r'(<script id="payload" type="application/json">).*?(</script>)', re.S)
html2 = pat.sub(lambda m: m.group(1) + data + m.group(2), html, count=1)
(ROOT / "dashboard/index.html").write_text(html2, encoding="utf-8")
print(f"re-inyectado: {len(html2)//1024} KB | seasonality={'seasonality' in data} neighbors={'neighbors' in data}")
