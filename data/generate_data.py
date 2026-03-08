"""
AEGIS AI - Synthetic Hiring Dataset Generator
Generates a biased hiring dataset for testing the ethical AI auditing framework.
"""

import pandas as pd
import numpy as np
import os

def generate_hiring_data(n_samples=2000, random_state=42):
    """
    Generate a synthetic hiring dataset with intentional biases for auditing.
    
    Biases introduced:
    - Gender bias: Males get slightly higher scores
    - Race bias: Certain races get preferential treatment
    - Age bias: Younger candidates preferred
    """
    np.random.seed(random_state)
    
    # Demographics
    genders = np.random.choice(['Male', 'Female', 'Non-Binary'], n_samples, p=[0.45, 0.45, 0.10])
    races = np.random.choice(
        ['White', 'Black', 'Asian', 'Hispanic', 'Other'],
        n_samples,
        p=[0.40, 0.20, 0.15, 0.15, 0.10]
    )
    ages = np.random.randint(22, 60, n_samples)
    
    # Qualifications (unbiased ground truth)
    education_years = np.random.randint(12, 22, n_samples)
    experience_years = np.clip(ages - 22 + np.random.randint(-2, 5, n_samples), 0, 35)
    skill_score = np.random.uniform(40, 100, n_samples)
    interview_score = np.random.uniform(30, 100, n_samples)
    certification_count = np.random.randint(0, 8, n_samples)
    project_count = np.random.randint(0, 15, n_samples)
    gpa = np.round(np.random.uniform(2.0, 4.0, n_samples), 2)
    
    # Create base hiring score (merit-based)
    base_score = (
        0.20 * (education_years / 22) * 100 +
        0.20 * (experience_years / 35) * 100 +
        0.25 * skill_score +
        0.20 * interview_score +
        0.05 * (certification_count / 8) * 100 +
        0.05 * (project_count / 15) * 100 +
        0.05 * (gpa / 4.0) * 100
    )
    
    # Introduce BIAS (this is what our system should detect!)
    bias_score = base_score.copy()
    
    # Gender bias: +5 points for males
    gender_bias = np.where(genders == 'Male', 5.0, 
                  np.where(genders == 'Female', -3.0, -1.0))
    bias_score += gender_bias
    
    # Race bias: +4 for White, -3 for Black, -2 for Hispanic
    race_bias = np.where(races == 'White', 4.0,
                np.where(races == 'Black', -3.0,
                np.where(races == 'Hispanic', -2.0, 0.0)))
    bias_score += race_bias
    
    # Age bias: penalty for older candidates
    age_bias = np.where(ages > 45, -4.0, np.where(ages > 35, -1.5, 1.0))
    bias_score += age_bias
    
    # Add noise
    noise = np.random.normal(0, 3, n_samples)
    bias_score += noise
    
    # Hiring decision (biased)
    threshold = np.percentile(bias_score, 60)  # Top 40% get hired
    hired = (bias_score >= threshold).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'candidate_id': range(1, n_samples + 1),
        'gender': genders,
        'race': races,
        'age': ages,
        'education_years': education_years,
        'experience_years': experience_years,
        'skill_score': np.round(skill_score, 2),
        'interview_score': np.round(interview_score, 2),
        'certification_count': certification_count,
        'project_count': project_count,
        'gpa': gpa,
        'hired': hired
    })
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'synthetic_hiring_data.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {n_samples} candidates → {output_path}")
    print(f"   Hired: {hired.sum()} ({hired.mean()*100:.1f}%)")
    print(f"   Gender distribution: {dict(zip(*np.unique(genders, return_counts=True)))}")
    print(f"   Race distribution: {dict(zip(*np.unique(races, return_counts=True)))}")
    
    return df


if __name__ == "__main__":
    df = generate_hiring_data()
    print("\nSample data:")
    print(df.head(10).to_string())
