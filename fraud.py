import pandas as pd

# قراءة الداتا سيت الأساسية
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# التأكد من خلو الداتا من أي قيم مفقودة وحذفها لو موجودة
df = df.dropna()

# تحويل عمود النوع (type) لأرقام عشان الموديل يفهمه
if 'type' in df.columns:
    df['type'] = df['type'].astype('category').cat.codes

# حفظ الداتا سيت بعد التعديل والتنظيف في ملف جديد للتيم
output_file = 'cleaned_fraud_data.csv'
df.to_csv(output_file, index=False)

print("تم تنظيف وحفظ الداتا بنجاح ن!")