from src.predict import get_pl_standings_from_api

rows = get_pl_standings_from_api(2025)
if rows:
    print(f"{'#':<4} {'Club':<25} {'MP':<5} {'W':<4} {'D':<4} {'L':<4} {'GF':<5} {'GA':<5} {'GD':<5} {'PTS'}")
    print("-" * 70)
    for r in rows:
        print(f"{r.get('pos', r.get('Club','?')):<4} {r['Club']:<25} {r['MP']:<5} {r['W']:<4} {r['D']:<4} {r['L']:<4} {r['GF']:<5} {r['GA']:<5} {r['GD']:<5} {r['PTS']}")
else:
    print("FAILED - ดึงข้อมูลไม่ได้")
