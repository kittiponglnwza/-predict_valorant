"""
Debug script — ดู HTML จาก Understat ว่า data ซ่อนอยู่ที่ไหน
รันแล้ว copy output มาให้ดู
"""
import requests
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

print("Fetching Understat EPL 2024...")
resp = requests.get("https://understat.com/league/EPL/2024", headers=HEADERS, timeout=20)
print(f"Status: {resp.status_code}")
print(f"Content-Length: {len(resp.text)}")

html = resp.text

# หา var ทั้งหมดที่ขึ้นต้นด้วย var ...Data
print("\n=== All 'var ...Data' variables found ===")
patterns = re.findall(r'var (\w+)\s*=\s*JSON\.parse', html)
print("Variables:", patterns)

# ลอง patterns ต่างๆ
print("\n=== Trying different patterns ===")
test_patterns = [
    r"var datesData\s*=\s*JSON\.parse\('(.+?)'\)",
    r'var datesData\s*=\s*JSON\.parse\("(.+?)"\)',
    r"datesData\s*=\s*JSON\.parse\('(.+?)'\)",
    r'JSON\.parse\(\'(.{100,}?)\'\)',
]
for p in test_patterns:
    m = re.search(p, html)
    print(f"Pattern: {p[:50]}...")
    print(f"  Found: {bool(m)}")
    if m:
        print(f"  First 100 chars: {m.group(1)[:100]}")

# แสดง raw HTML รอบๆ คำว่า datesData
print("\n=== HTML around 'datesData' ===")
idx = html.find('datesData')
if idx >= 0:
    print(repr(html[max(0,idx-20):idx+200]))
else:
    print("'datesData' NOT FOUND in HTML")

# ลองหา JSON pattern อื่น
print("\n=== Looking for match data patterns ===")
for keyword in ['datesData', 'teamsData', 'playersData', 'JSON.parse', 'isResult']:
    idx = html.find(keyword)
    if idx >= 0:
        print(f"Found '{keyword}' at position {idx}")
        print(f"  Context: {repr(html[max(0,idx-10):idx+100])}")
    else:
        print(f"'{keyword}' NOT FOUND")
