import pickle, sys, os
sys.path.insert(0, '.')

path = 'models/football_model_v9.pkl'
if not os.path.exists(path):
    print("❌ ไม่พบไฟล์:", path)
    sys.exit()

ctx = pickle.load(open(path, 'rb'))

rf = ctx.get('remaining_fixtures', [])
ft = ctx.get('final_table')

print("remaining_fixtures :", len(rf))
print("final_table        :", ft is not None)

if len(rf) > 0:
    print("ตัวอย่าง fixture    :", rf[0])
else:
    print("⚠️  remaining_fixtures ว่าง — Monte Carlo จะ return None")
    print("   ต้องรัน run_season_simulation() ก่อน หรือ set remaining_fixtures ใน ctx")
