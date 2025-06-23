"""
Example Usage of EM MDM Extraction System
Demonstrates how to use the workflow for extracting MDM levels from patient records
"""

import json
import os
from workflow import EMMDMWorkflow
from em_mdm_extractor import EMMDMExtractor

def example_1_basic_usage():
    """Example 1: Basic usage with simple patient record"""
    print("=== Example 1: Basic Usage ===")
    
    # Sample patient record
    patient_record = """
    CHIEF COMPLAINT: Fever and cough
    
    HPI: 45-year-old female presents with 2-day history of fever (101.5°F) and dry cough.
    No chest pain, shortness of breath, or other symptoms. No recent travel or sick contacts.
    
    ASSESSMENT:
    1. Upper respiratory infection
    2. Fever
    
    PLAN:
    - Tylenol for fever
    - Rest and fluids
    - Return if symptoms worsen
    
    MEDICATIONS:
    - Tylenol 500mg PRN (OTC)
    """
    
    try:
        # Initialize extractor
        extractor = EMMDMExtractor()
        
        # Process patient record
        result = extractor.process_patient_record(patient_record)
        
        print(f"Problems Level: {result.problems_level.value}")
        print(f"Data Level: {result.data_level.value}")
        print(f"Risk Level: {result.risk_level.value}")
        print(f"Final MDM Level: {result.final_level.value}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Reasoning: {result.reasoning}")
        
    except Exception as e:
        print(f"Error: {e}")

def example_2_complex_case():
    """Example 2: Complex case with multiple problems and interventions"""
    print("\n=== Example 2: Complex Case ===")
    
    patient_record = """
    CHIEF COMPLAINT: Severe chest pain and shortness of breath
    
    HPI: 72-year-old male with known CAD, diabetes, and hypertension presents with severe 
    substernal chest pain radiating to left arm, associated with diaphoresis and nausea. 
    Pain started 2 hours ago and is unrelieved by nitroglycerin. Patient has history of 
    previous MI and CABG 3 years ago.
    
    ASSESSMENT:
    1. Acute ST-elevation myocardial infarction
    2. Uncontrolled hypertension
    3. Diabetes mellitus type 2 with complications
    4. Congestive heart failure exacerbation
    5. Acute kidney injury
    
    PLAN:
    - Immediate cardiac catheterization
    - Aspirin 325mg, Plavix 600mg, Heparin bolus
    - Nitroglycerin drip
    - Metoprolol 5mg IV
    - Cardiology consultation
    - Cardiothoracic surgery consultation
    
    LAB ORDERS:
    - Troponin I (baseline and serial)
    - CBC, CMP, PT/INR, PTT
    - BNP, CK-MB
    - Lipid panel
    - HbA1c
    
    IMAGING:
    - ECG: ST elevation anterior leads
    - Chest X-ray: Pulmonary edema
    - Echocardiogram: LVEF 25%
    
    DOCUMENTS REVIEWED:
    - Previous cardiology consultation
    - Recent stress test results
    - Medication reconciliation from pharmacy
    
    MEDICATIONS:
    - Aspirin 325mg (new)
    - Plavix 600mg (new)
    - Heparin drip (new)
    - Nitroglycerin drip (new)
    - Metoprolol 5mg IV (new)
    - Lisinopril 10mg daily (continued)
    - Metformin 1000mg BID (continued)
    - Insulin sliding scale (continued)
    
    RISK FACTORS:
    - Previous MI
    - Previous CABG
    - Diabetes with complications
    - Hypertension
    - Age >70
    - Male gender
    - Family history of CAD
    """
    
    try:
        # Initialize workflow
        workflow = EMMDMWorkflow()
        
        # Execute complete workflow
        result = workflow.execute_workflow(patient_record, "COMPLEX_PATIENT_001")
        
        # Generate and print report
        report = workflow.generate_workflow_report(result)
        print(report)
        
    except Exception as e:
        print(f"Error: {e}")

def example_3_batch_processing():
    """Example 3: Batch processing multiple patient records"""
    print("\n=== Example 3: Batch Processing ===")
    
    # Multiple patient records
    patient_records = [
        {
            "id": "PATIENT_001",
            "record": """
            CHIEF COMPLAINT: Sore throat
            
            HPI: 25-year-old female with 3-day history of sore throat, difficulty swallowing.
            No fever, no cough. No recent travel.
            
            ASSESSMENT: Pharyngitis
            
            PLAN: Amoxicillin 500mg TID x 10 days
            
            MEDICATIONS: Amoxicillin 500mg TID (new)
            """
        },
        {
            "id": "PATIENT_002", 
            "record": """
            CHIEF COMPLAINT: Abdominal pain
            
            HPI: 35-year-old male with severe right lower quadrant pain for 6 hours.
            Associated with nausea and vomiting. No fever.
            
            ASSESSMENT: Acute appendicitis
            
            PLAN: 
            - Emergency appendectomy
            - Pre-op labs: CBC, CMP, PT/INR
            - Surgery consultation
            
            MEDICATIONS: 
            - Morphine 2mg IV (new)
            - Cefazolin 1g IV (new)
            
            LAB ORDERS: CBC, CMP, PT/INR
            """
        },
        {
            "id": "PATIENT_003",
            "record": """
            CHIEF COMPLAINT: Follow-up for diabetes
            
            HPI: 60-year-old female with type 2 diabetes for 10 years.
            Blood sugar well controlled on current medications.
            No new symptoms or concerns.
            
            ASSESSMENT: Diabetes mellitus type 2, stable
            
            PLAN: Continue current medications, monitor blood sugar
            
            MEDICATIONS: Metformin 1000mg BID (continued)
            
            LAB ORDERS: HbA1c, CMP
            """
        }
    ]
    
    try:
        # Initialize extractor
        extractor = EMMDMExtractor()
        
        results = []
        
        for patient in patient_records:
            print(f"\nProcessing {patient['id']}...")
            
            result = extractor.process_patient_record(patient['record'])
            
            results.append({
                "patient_id": patient['id'],
                "problems_level": result.problems_level.value,
                "data_level": result.data_level.value,
                "risk_level": result.risk_level.value,
                "final_level": result.final_level.value,
                "confidence": result.confidence_score
            })
            
            print(f"  Final MDM Level: {result.final_level.value}")
            print(f"  Confidence: {result.confidence_score:.2f}")
        
        # Save batch results
        with open("batch_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBatch processing completed. Results saved to batch_results.json")
        
        # Summary statistics
        levels = [r['final_level'] for r in results]
        print(f"\nSummary:")
        print(f"  Total patients: {len(results)}")
        print(f"  Low complexity: {levels.count('Low')}")
        print(f"  Moderate complexity: {levels.count('Moderate')}")
        print(f"  High complexity: {levels.count('High')}")
        
    except Exception as e:
        print(f"Error: {e}")

def example_4_detailed_analysis():
    """Example 4: Detailed analysis with all intermediate results"""
    print("\n=== Example 4: Detailed Analysis ===")
    
    patient_record = """
    CHIEF COMPLAINT: Syncope and chest pain
    
    HPI: 68-year-old male with history of CAD, hypertension, and diabetes presents with 
    episode of syncope this morning followed by chest pain. Patient was found on floor 
    by family member. No head trauma. Chest pain is substernal, pressure-like, radiating 
    to jaw. Associated with diaphoresis and shortness of breath.
    
    ASSESSMENT:
    1. Syncope - rule out cardiac cause
    2. Acute coronary syndrome
    3. Hypertension - uncontrolled
    4. Diabetes mellitus type 2
    5. Arrhythmia - rule out
    
    PLAN:
    - Cardiac monitoring
    - ECG: Normal sinus rhythm, no acute changes
    - Troponin I: 0.8 ng/mL (elevated)
    - CBC: Normal
    - CMP: Glucose 220, BUN 30, Creatinine 1.4
    - Chest X-ray: Normal
    - Echocardiogram: LVEF 45%, no wall motion abnormalities
    - Cardiology consultation
    - Neurology consultation for syncope workup
    
    DOCUMENTS REVIEWED:
    - Previous cardiology notes
    - Recent stress test (normal)
    - Medication list from pharmacy
    - Family member statement
    
    MEDICATIONS:
    - Aspirin 325mg (new)
    - Metoprolol 25mg BID (new)
    - Lisinopril 10mg daily (continued)
    - Metformin 1000mg BID (continued)
    - Atorvastatin 40mg daily (continued)
    
    RISK FACTORS:
    - Age >65
    - Previous CAD
    - Hypertension
    - Diabetes
    - Male gender
    """
    
    try:
        # Initialize extractor
        extractor = EMMDMExtractor()
        
        # Get detailed analysis
        detailed = extractor.get_detailed_analysis(patient_record)
        
        print("=== DETAILED EXTRACTION RESULTS ===")
        print(json.dumps(detailed['extraction_results'], indent=2))
        
        print("\n=== MATCHING RESULTS ===")
        print(json.dumps(detailed['matching_results'], indent=2))
        
        print("\n=== FINAL EVALUATION ===")
        print(json.dumps(detailed['final_evaluation'], indent=2))
        
        print("\n=== VALIDATION ===")
        print(json.dumps(detailed['validation'], indent=2))
        
        print("\n=== SUMMARY ===")
        print(json.dumps(detailed['summary'], indent=2))
        
        # Save detailed results
        with open("detailed_analysis.json", "w") as f:
            json.dump(detailed, f, indent=2)
        
        print(f"\nDetailed analysis saved to detailed_analysis.json")
        
    except Exception as e:
        print(f"Error: {e}")

def example_5_validation_focus():
    """Example 5: Focus on validation and cross-validation"""
    print("\n=== Example 5: Validation Focus ===")
    
    patient_record = """
    CHIEF COMPLAINT: Follow-up for hypertension
    
    HPI: 55-year-old female with hypertension for 5 years. Blood pressure well controlled 
    on current medications. No new symptoms or concerns. Last visit 3 months ago.
    
    ASSESSMENT: Hypertension, controlled
    
    PLAN: Continue current medications, monitor blood pressure
    
    LAB ORDERS: CMP, Lipid panel
    
    MEDICATIONS: Lisinopril 10mg daily (continued)
    
    DOCUMENTS REVIEWED: Previous visit note
    """
    
    try:
        # Initialize workflow
        workflow = EMMDMWorkflow()
        
        # Execute workflow
        result = workflow.execute_workflow(patient_record, "VALIDATION_TEST_001")
        
        print("=== VALIDATION RESULTS ===")
        print(f"Validation Passed: {result.validation_passed}")
        print(f"Cross-Validation Passed: {result.cross_validation_passed}")
        print(f"Overall Confidence: {result.overall_confidence:.2f}")
        
        # Check validation step specifically
        validation_step = next((s for s in result.steps if s.step_name == "Validation"), None)
        if validation_step:
            print(f"\nValidation Step Details:")
            print(f"  Confidence: {validation_step.confidence_score:.2f}")
            print(f"  Issues: {validation_step.output_data.get('issues_identified', [])}")
            print(f"  Recommendations: {validation_step.output_data.get('recommendations', [])}")
        
        # Check cross-validation step
        cross_validation_step = next((s for s in result.steps if s.step_name == "Cross-Validation"), None)
        if cross_validation_step:
            print(f"\nCross-Validation Step Details:")
            print(f"  Confidence: {cross_validation_step.confidence_score:.2f}")
            print(f"  Contradictions: {cross_validation_step.output_data.get('contradictions', [])}")
            print(f"  Clinical Validation: {cross_validation_step.output_data.get('clinical_validation', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all examples"""
    print("EM MDM Extraction System - Example Usage")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key before running examples.")
        return
    
    # Run examples
    example_1_basic_usage()
    example_2_complex_case()
    example_3_batch_processing()
    example_4_detailed_analysis()
    example_5_validation_focus()
    
    print("\n" + "=" * 50)
    print("All examples completed!")

if __name__ == "__main__":
    main() 