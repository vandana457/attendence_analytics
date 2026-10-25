import pandas as pd
import numpy as np

# -------------------------------------------------------------
# Section 0: Data Loading
# -------------------------------------------------------------
# Load Excel dataset
excel_file = "attendance.xlsx"  # Change to "aids_attendance.xlsx" if needed
df = pd.read_excel(excel_file)

# -------------------------------------------------------------
# Section 1: Data Inspection
# -------------------------------------------------------------
# Q1. First 10 rows, shape, and dtypes
print("=== Q1 ===")
print(df.head(10))
print("\nShape:", df.shape)
print("\nData Types:\n", df.dtypes)

# Q2. Check for missing values & 'Never' in Last Present Date
print("\n=== Q2 ===")
print("Missing values:\n", df.isnull().sum())
never_count = (df['Last Present Date'] == 'Never').sum()
print("Count of 'Never' in Last Present Date:", never_count)

# Q3. Rename columns to snake_case
print("\n=== Q3 ===")
rename_dict = {
    'Roll No': 'roll_no',
    'Student Name': 'student_name',
    'Subject': 'subject',
    'Total Classes': 'total_classes',
    'Present': 'present',
    'Absent': 'absent',
    'Excused': 'excused',
    'Attendance %': 'attendance_pct',
    'Status': 'status',
    'Classes Needed (75%)': 'classes_needed_75',
    'Last Present Date': 'last_present_date',
    'Consecutive Absences': 'consecutive_absences'
}
df = df.rename(columns=rename_dict)
print("Updated Columns:", df.columns.tolist())

# Q4. Convert Attendance % to numeric float and Last Present Date to datetime
print("\n=== Q4 ===")
df['attendance_pct_num'] = df['attendance_pct'].astype(str).str.rstrip('%').astype(float)
df['last_present_date_dt'] = pd.to_datetime(
    df['last_present_date'].replace('Never', np.nan), 
    errors='coerce'
)

# -------------------------------------------------------------
# Section 2: Filtering & Selection
# -------------------------------------------------------------
# Q5. Critical status students sorted by Attendance %
print("\n=== Q5 ===")
critical_sorted = df[df['status'] == 'Critical'].sort_values(by='attendance_pct_num', ascending=True)
print(critical_sorted[['roll_no', 'student_name', 'attendance_pct_num', 'status']])

# Q6. Students with 0 present days
print("\n=== Q6 ===")
zero_present = df[df['present'] == 0][['roll_no', 'student_name', 'consecutive_absences']]
print(zero_present)

# Q7. Using .loc and .iloc
print("\n=== Q7 ===")
safe_students = df.loc[df['status'] == 'Safe']
iloc_subset = df.iloc[0:15, 2:6]
print("Safe Students Count:", len(safe_students))
print("iloc Subset Preview:\n", iloc_subset.head())

# Q8. Consecutive Absences >= 5
print("\n=== Q8 ===")
consec_5_plus = df[df['consecutive_absences'] >= 5][['student_name', 'status', 'last_present_date']]
print(consec_5_plus)

# -------------------------------------------------------------
# Section 3: Grouping & Aggregation
# -------------------------------------------------------------
# Q9. Group by Status metrics
print("\n=== Q9 ===")
q9_summary = df.groupby('status').agg(
    student_count=('roll_no', 'count'),
    avg_attendance=('attendance_pct_num', 'mean'),
    max_consecutive_absences=('consecutive_absences', 'max')
)
print(q9_summary)

# Q10. Attendance Category
print("\n=== Q10 ===")
def categorize_attendance(pct):
    if pct >= 90:
        return 'Excellent'
    elif pct >= 75:
        return 'Good'
    elif pct >= 60:
        return 'Average'
    else:
        return 'Poor'

df['attendance_category'] = df['attendance_pct_num'].apply(categorize_attendance)
print(df.groupby('attendance_category').size())

# Q11. Total class counts
print("\n=== Q11 ===")
print(df[['total_classes', 'present', 'absent']].agg('sum'))

# Q12. Aggregations on Present and Absent
print("\n=== Q12 ===")
q12_summary = df.groupby('status').agg(
    min_present=('present', 'min'),
    max_present=('present', 'max'),
    mean_present=('present', 'mean'),
    sum_absent=('absent', 'sum')
)
print(q12_summary)

# -------------------------------------------------------------
# Section 4: Feature Engineering & Transformation
# -------------------------------------------------------------
# Q13. Risk Score calculation
print("\n=== Q13 ===")
df['risk_score'] = (df['absent'] * 0.4) + (df['consecutive_absences'] * 0.6)
top_10_risk = df.sort_values(by='risk_score', ascending=False)[
    ['roll_no', 'student_name', 'absent', 'consecutive_absences', 'risk_score']
].head(10)
print(top_10_risk)

# Q14. Extract numeric value from Classes Needed
print("\n=== Q14 ===")
df['classes_needed_numeric'] = df['classes_needed_75'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)

# Q15. Attendance Flag using apply & lambda
print("\n=== Q15 ===")
df['attendance_flag'] = df['attendance_pct_num'].apply(lambda x: 'Defaulter' if x < 75 else 'Safe')
print(df['attendance_flag'].value_counts())

# Q16. Binning attendance with pd.cut()
print("\n=== Q16 ===")
bins = [0, 50, 75, 90, 100]
labels = ["Critical", "Warning", "Good", "Excellent"]
df['binned_attendance'] = pd.cut(df['attendance_pct_num'], bins=bins, labels=labels, include_lowest=True)
print(df['binned_attendance'].value_counts())

# -------------------------------------------------------------
# Section 5: Advanced Analysis
# -------------------------------------------------------------
# Q17. Pivot table by Status and Attendance Category
print("\n=== Q17 ===")
pivot_summary = pd.pivot_table(
    df, index='status', columns='attendance_category', 
    values='roll_no', aggfunc='count', fill_value=0
)
print(pivot_summary)

# Q18. Duplicate student base names
print("\n=== Q18 ===")
df['base_name'] = df['student_name'].astype(str).str.split().str[0]
duplicated_students = df[df.duplicated(subset=['base_name'], keep=False)].sort_values('base_name')
print(duplicated_students[['roll_no', 'student_name', 'base_name']])

# Q19. Merge with hypothetical Student_Details DataFrame
print("\n=== Q19 ===")
student_details = pd.DataFrame({
    'roll_no': df['roll_no'],
    'branch': 'AI&DS',
    'semester': 5
})
merged_df = pd.merge(df, student_details, on='roll_no', how='inner')
print(merged_df[['roll_no', 'student_name', 'branch', 'semester']].head())

# Q20. Export CSV files
print("\n=== Q20 ===")
df[df['status'] == 'Critical'].to_csv('critical_students.csv', index=False)
df.groupby('status').describe().to_csv('status_summary.csv')
df.sort_values(by='risk_score', ascending=False).head(10).to_csv('top10_risk_students.csv', index=False)
print("All reports generated and exported to CSV successfully!")