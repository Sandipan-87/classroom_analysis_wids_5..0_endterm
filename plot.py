import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
COLORS = {
    "Focused": "#2ecc71",       
    "Taking Notes": "#3498db", 
    "Looking Away": "#f1c40f",  
    "Distracted": "#e74c3c"     
}

def find_latest_report():
    files = glob.glob("attention_report_*.xlsx")
    if not files:
        print(" No report files found!")
        print("   Run main.py first to generate an Excel report.")
        sys.exit(1)
    return max(files, key=os.path.getctime)

def plot_timeline(df_timeline):
    """Generates the timeline using clean lines instead of stacked areas."""
    plt.figure(figsize=(14, 12))  # Slightly larger for better visibility
    
    # ==========================================
    # PLOT 1: Student States 
    # ==========================================
    plt.subplot(2, 1, 1)
    
    # Plot each state as a separate distinct line
    for state in ["Focused", "Taking Notes", "Looking Away", "Distracted"]:
        if state in df_timeline.columns:
            sns.lineplot(
                x=df_timeline["Time"], 
                y=df_timeline[state], 
                label=state, 
                color=COLORS[state], 
                linewidth=3.5, 
                alpha=1.0       
            )

    plt.title("Classroom State Counts Over Time", fontsize=16, fontweight='bold')
    plt.ylabel("Number of Students", fontsize=14)
    plt.xlabel("") 
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Legend Settings
    plt.legend(
        loc="upper left", 
        frameon=True, 
        fontsize=12, 
        framealpha=0.9, 
        edgecolor="#333"
    )
    plt.xlim(df_timeline["Time"].min(), df_timeline["Time"].max())

    # ==========================================
    # PLOT 2: Focus Index (Trend)
    # ==========================================
    plt.subplot(2, 1, 2)
    sns.lineplot(
        data=df_timeline, 
        x="Time", 
        y="Class Focus %", 
        color="#2c3e50", 
        linewidth=4 
    )
    
    # Add Reference Line
    plt.axhline(70, color="#27ae60", linestyle="--", linewidth=2, alpha=0.8, label="Target (70%)")
    
    # Fill under the curve
    plt.fill_between(
        df_timeline["Time"], 
        df_timeline["Class Focus %"], 
        alpha=0.15, 
        color="#2c3e50"
    )
    
    plt.title("Class Focus Index Trend", fontsize=16, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("Focus Index (%)", fontsize=14)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower right", fontsize=12)
    plt.xlim(df_timeline["Time"].min(), df_timeline["Time"].max())

    plt.tight_layout()
    plt.savefig("plot_timeline.png", dpi=300)
    print(" Saved plot_timeline.png (High Visibility)")
    plt.show()

def plot_student_performance(df_students):
    """Generates bar charts for individual student performance."""
    if df_students.empty:
        print(" No student data to plot.")
        return

    # Sort by Focus Index
    df_students = df_students.sort_values("Focus Index", ascending=True)

    plt.figure(figsize=(12, max(6, len(df_students) * 0.6)))
    
    # Color bars based on score
    bar_colors = [
        COLORS["Focused"] if x >= 70 else 
        COLORS["Looking Away"] if x >= 50 else 
        COLORS["Distracted"] 
        for x in df_students["Focus Index"]
    ]
    
    bars = plt.barh(df_students["Student ID"], df_students["Focus Index"], color=bar_colors, height=0.7)
    
    plt.axvline(70, color="green", linestyle="--", alpha=0.5, label="Goal (70%)")
    plt.title("Student Focus Index Ranking", fontsize=16, fontweight='bold')
    plt.xlabel("Focus Index Score (0-100)", fontsize=12)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.0f}', va='center', fontweight='bold', fontsize=10)

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_students.png", dpi=300)
    print(" Saved plot_students.png")
    plt.show()

def plot_overall_pie(df_timeline):
    """Generates a pie chart of total time spent in each state."""
    totals = {
        "Focused": df_timeline["Focused"].sum(),
        "Taking Notes": df_timeline["Taking Notes"].sum(),
        "Looking Away": df_timeline["Looking Away"].sum(),
        "Distracted": df_timeline["Distracted"].sum()
    }
    
    totals = {k: v for k, v in totals.items() if v > 0}
    
    plt.figure(figsize=(9, 9))
    plt.pie(
        totals.values(), 
        labels=totals.keys(),
        colors=[COLORS[k] for k in totals.keys()],
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 13, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5}
    )
    plt.title("Overall Session State Distribution", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plot_distribution.png", dpi=300)
    print("✅ Saved plot_distribution.png")
    plt.show()

def main():
    file_path = find_latest_report()
    print(f" Analyzing: {file_path}")
    
    try:
        df_timeline = pd.read_excel(file_path, sheet_name="Timeline")
        df_students = pd.read_excel(file_path, sheet_name="Students")
    except Exception as e:
        print(f" Error reading Excel file: {e}")
        return

    sns.set_theme(style="whitegrid")
    
    print("Generating Timeline Plot...")
    plot_timeline(df_timeline)
    
    print("Generating Student Ranking...")
    plot_student_performance(df_students)
    
    print("Generating Overall Distribution...")
    plot_overall_pie(df_timeline)

if __name__ == "__main__":

    main()
