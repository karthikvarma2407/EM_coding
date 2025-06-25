"""
EM MDM Extraction Prompts
Contains all prompts used in the EM MDM extraction workflow
"""

class EMMDMPrompts:
    """Collection of all prompts for EM MDM extraction"""
    
    @staticmethod
    def get_problems_extraction_prompt() -> str:
        """Prompt for extracting information relevant to Problems table"""
        return """
You are an expert medical coder specializing in Evaluation and Management (E/M) coding using the 2023 AMA guidelines.

TASK: Extract information from the patient medical record that is relevant for determining the NUMBER OF DIAGNOSES OR MANAGEMENT OPTIONS in the Problems table of Medical Decision Making (MDM).

CONTEXT: The Problems table evaluates the number and complexity of problems addressed during the encounter.

GUIDELINES FOR PROBLEMS TABLE:
- Minimal: 1 problem
- Low: 2 problems OR 1 stable chronic illness OR 1 acute uncomplicated illness/injury
- Moderate: 3 problems OR 1 chronic illness with exacerbation/progression OR 1 acute complicated injury
- High: 4+ problems OR 1 chronic illness with severe exacerbation/progression OR 1 acute/chronic illness posing threat to life/body function

EXTRACT THE FOLLOWING INFORMATION:
1. Number of distinct problems addressed
2. Type of each problem (acute, chronic, stable, unstable)
3. Severity indicators (exacerbation, progression, complications)
4. Life-threatening conditions
5. Body function threats
6. Management complexity indicators

PATIENT RECORD:
{patient_record}

Provide your analysis in the following JSON format:
{
    "problems_count": number,
    "problem_types": ["acute", "chronic", "stable", "unstable"],
    "severity_indicators": ["exacerbation", "progression", "complications"],
    "life_threatening": boolean,
    "body_function_threat": boolean,
    "management_complexity": ["low", "moderate", "high"],
    "extracted_details": "detailed explanation of findings"
}
"""

    @staticmethod
    def get_problems_matching_prompt() -> str:
        """Prompt for matching extracted information to Problems table criteria"""
        return """
You are an expert medical coder evaluating the Problems table for Medical Decision Making (MDM).

TASK: Match the extracted information to the specific criteria in the Problems table to determine the appropriate level.

EXTRACTED INFORMATION:
{extracted_info}

PROBLEMS TABLE CRITERIA:
- Minimal: 1 problem
- Low: 2 problems OR 1 stable chronic illness OR 1 acute uncomplicated illness/injury
- Moderate: 3 problems OR 1 chronic illness with exacerbation/progression OR 1 acute complicated injury
- High: 4+ problems OR 1 chronic illness with severe exacerbation/progression OR 1 acute/chronic illness posing threat to life/body function

ANALYZE AND PROVIDE:
1. Which criteria match the extracted information
2. The predicted MDM level for Problems
3. Confidence score (0-1)
4. Detailed reasoning

Respond in JSON format:
{
    "matched_criteria": ["list of matching criteria"],
    "predicted_level": "Minimal/Low/Moderate/High",
    "confidence_score": 0.95,
    "reasoning": "detailed explanation"
}
"""

    @staticmethod
    def get_data_extraction_prompt() -> str:
        """Prompt for extracting information relevant to Data table"""
        return """
You are an expert medical coder specializing in Evaluation and Management (E/M) coding using the 2023 AMA guidelines.

TASK: Extract information from the patient medical record that is relevant for determining the AMOUNT AND/OR COMPLEXITY OF DATA TO BE REVIEWED AND ANALYZED in the Data table of Medical Decision Making (MDM).

CONTEXT: The Data table evaluates the amount and complexity of data reviewed during the encounter.

GUIDELINES FOR DATA TABLE:
- Minimal: No data reviewed
- Low: 1 test/lab OR 1 document review OR 1 independent historian
- Moderate: 2 tests/labs OR 1 test + 1 document OR 1 test + 1 independent historian OR 2 documents OR 2 independent historians
- High: 3+ tests/labs OR 2+ tests + 1 document OR 1 test + 2+ documents OR 3+ documents OR 3+ independent historians

EXTRACT THE FOLLOWING INFORMATION:
1. Number and types of lab tests ordered/reviewed
2. Number and types of imaging studies
3. Documents reviewed (consultations, reports, etc.)
4. Independent historians consulted
5. Data complexity indicators
6. Clinical decision support tools used

PATIENT RECORD:
{patient_record}

Provide your analysis in the following JSON format:
{
    "lab_tests": ["list of lab tests"],
    "imaging_studies": ["list of imaging studies"],
    "documents_reviewed": ["list of documents"],
    "independent_historians": ["list of historians"],
    "clinical_decision_support": ["list of tools used"],
    "data_complexity": ["low", "moderate", "high"],
    "extracted_details": "detailed explanation of findings"
}
"""

    @staticmethod
    def get_data_matching_prompt() -> str:
        """Prompt for matching extracted information to Data table criteria"""
        return """
You are an expert medical coder evaluating the Data table for Medical Decision Making (MDM).

TASK: Match the extracted information to the specific criteria in the Data table to determine the appropriate level.

EXTRACTED INFORMATION:
{extracted_info}

DATA TABLE CRITERIA:
- Minimal: No data reviewed
- Low: 1 test/lab OR 1 document review OR 1 independent historian
- Moderate: 2 tests/labs OR 1 test + 1 document OR 1 test + 1 independent historian OR 2 documents OR 2 independent historians
- High: 3+ tests/labs OR 2+ tests + 1 document OR 1 test + 2+ documents OR 3+ documents OR 3+ independent historians

ANALYZE AND PROVIDE:
1. Which criteria match the extracted information
2. The predicted MDM level for Data
3. Confidence score (0-1)
4. Detailed reasoning

Respond in JSON format:
{
    "matched_criteria": ["list of matching criteria"],
    "predicted_level": "Minimal/Low/Moderate/High",
    "confidence_score": 0.95,
    "reasoning": "detailed explanation"
}
"""

    @staticmethod
    def get_risk_extraction_prompt() -> str:
        """Prompt for extracting information relevant to Risk table"""
        return """
You are an expert medical coder specializing in Evaluation and Management (E/M) coding using the 2023 AMA guidelines.

TASK: Extract information from the patient medical record that is relevant for determining the RISK OF COMPLICATIONS AND/OR MORBIDITY OR MORTALITY in the Risk table of Medical Decision Making (MDM).

CONTEXT: The Risk table evaluates the risk of complications, morbidity, or mortality associated with the patient's condition and management.

GUIDELINES FOR RISK TABLE:
- Minimal: No risk
- Low: Prescription drug management OR over-the-counter drugs OR durable medical equipment OR social determinants of health
- Moderate: Minor surgery OR prescription drug management with monitoring OR social determinants of health with counseling OR diagnostic endoscopies
- High: Major surgery OR prescription drug management with intensive monitoring OR diagnostic endoscopies with identified risk factors OR social determinants of health with community resources

EXTRACT THE FOLLOWING INFORMATION:
1. Surgical procedures (minor/major)
2. Prescription drug management details
3. Over-the-counter drug use
4. Durable medical equipment
5. Social determinants of health
6. Diagnostic procedures
7. Risk factors identified
8. Monitoring requirements
9. Community resource utilization

PATIENT RECORD:
{patient_record}

Provide your analysis in the following JSON format:
{
    "surgical_procedures": ["list of procedures with type"],
    "prescription_management": ["details of prescription management"],
    "otc_medications": ["list of OTC medications"],
    "durable_equipment": ["list of DME"],
    "social_determinants": ["list of social determinants"],
    "diagnostic_procedures": ["list of diagnostic procedures"],
    "risk_factors": ["list of risk factors"],
    "monitoring_requirements": ["list of monitoring needs"],
    "community_resources": ["list of community resources"],
    "extracted_details": "detailed explanation of findings"
}
"""

    @staticmethod
    def get_risk_matching_prompt() -> str:
        """Prompt for matching extracted information to Risk table criteria"""
        return """
You are an expert medical coder evaluating the Risk table for Medical Decision Making (MDM).

TASK: Match the extracted information to the specific criteria in the Risk table to determine the appropriate level.

EXTRACTED INFORMATION:
{extracted_info}

RISK TABLE CRITERIA:
- Minimal: No risk
- Low: Prescription drug management OR over-the-counter drugs OR durable medical equipment OR social determinants of health
- Moderate: Minor surgery OR prescription drug management with monitoring OR social determinants of health with counseling OR diagnostic endoscopies
- High: Major surgery OR prescription drug management with intensive monitoring OR diagnostic endoscopies with identified risk factors OR social determinants of health with community resources

ANALYZE AND PROVIDE:
1. Which criteria match the extracted information
2. The predicted MDM level for Risk
3. Confidence score (0-1)
4. Detailed reasoning

Respond in JSON format:
{
    "matched_criteria": ["list of matching criteria"],
    "predicted_level": "Minimal/Low/Moderate/High",
    "confidence_score": 0.95,
    "reasoning": "detailed explanation"
}
"""

    @staticmethod
    def get_final_evaluation_prompt() -> str:
        """Prompt for final MDM level evaluation"""
        return """
You are an expert medical coder determining the final Medical Decision Making (MDM) level.

TASK: Evaluate the final MDM level based on the results from all three tables (Problems, Data, Risk).

MDM LEVEL DETERMINATION RULES:
- For 2 out of 3 elements meeting or exceeding a level, the overall MDM level is that level
- If 2 out of 3 elements are Moderate, the overall level is Moderate
- If 2 out of 3 elements are High, the overall level is High
- If 2 out of 3 elements are Low, the overall level is Low
- If 2 out of 3 elements are Minimal, the overall level is Minimal

TABLE RESULTS:
Problems: {problems_level} (Confidence: {problems_confidence})
Data: {data_level} (Confidence: {data_confidence})
Risk: {risk_level} (Confidence: {risk_confidence})

DETAILED ANALYSIS:
Problems Details: {problems_details}
Data Details: {data_details}
Risk Details: {risk_details}

DETERMINE:
1. Final MDM level
2. Overall confidence score
3. Detailed reasoning for the final level

Respond in JSON format:
{
    "final_level": "Minimal/Low/Moderate/High",
    "confidence_score": 0.95,
    "reasoning": "detailed explanation of final level determination",
    "table_breakdown": {
        "problems": "level and reasoning",
        "data": "level and reasoning", 
        "risk": "level and reasoning"
    }
}
"""

    @staticmethod
    def get_validation_prompt() -> str:
        """Prompt for validating the final MDM result"""
        return """
You are an expert medical coder performing final validation of MDM level determination.

TASK: Validate the MDM level determination for accuracy and consistency.

VALIDATION CRITERIA:
1. Check if the level determination follows AMA guidelines
2. Verify consistency across all three tables
3. Ensure the reasoning is sound and well-supported
4. Check for any potential errors or inconsistencies
5. Validate confidence scores are appropriate

MDM RESULT TO VALIDATE:
{mdm_result}

PATIENT RECORD SUMMARY:
{patient_summary}

PERFORM VALIDATION AND PROVIDE:
1. Is the result valid? (Yes/No)
2. Confidence in validation (0-1)
3. Any issues or concerns identified
4. Recommendations for improvement
5. Final validated MDM level

Respond in JSON format:
{
    "is_valid": true,
    "validation_confidence": 0.95,
    "issues_identified": ["list of any issues"],
    "recommendations": ["list of recommendations"],
    "final_validated_level": "Minimal/Low/Moderate/High",
    "validation_reasoning": "detailed explanation"
}
"""

    @staticmethod
    def get_enhanced_extraction_prompt() -> str:
        """Enhanced prompt for more detailed extraction"""
        return """
You are an expert medical coder with deep knowledge of the 2023 AMA E/M guidelines.

TASK: Perform comprehensive extraction of information from the patient medical record for MDM analysis.

CONTEXT: You need to extract detailed information that will be used to determine MDM levels across all three tables (Problems, Data, Risk).

EXTRACTION REQUIREMENTS:

1. PROBLEMS TABLE INFORMATION:
   - Count and categorize all problems addressed
   - Identify acute vs chronic conditions
   - Note exacerbations, progressions, complications
   - Identify life-threatening conditions
   - Assess management complexity

2. DATA TABLE INFORMATION:
   - List all lab tests ordered/reviewed
   - List all imaging studies
   - Identify documents reviewed
   - Note independent historians
   - Assess data complexity

3. RISK TABLE INFORMATION:
   - Identify surgical procedures
   - List prescription drug management
   - Note OTC medications
   - Identify DME
   - Assess social determinants
   - Note diagnostic procedures
   - Identify risk factors

PATIENT RECORD:
{patient_record}

Provide comprehensive extraction in JSON format:
{
    "problems_analysis": {
        "total_problems": number,
        "acute_problems": ["list"],
        "chronic_problems": ["list"],
        "exacerbations": ["list"],
        "complications": ["list"],
        "life_threatening": boolean,
        "management_complexity": "low/moderate/high"
    },
    "data_analysis": {
        "lab_tests": ["list"],
        "imaging_studies": ["list"],
        "documents": ["list"],
        "historians": ["list"],
        "total_data_points": number,
        "complexity": "low/moderate/high"
    },
    "risk_analysis": {
        "surgical_procedures": ["list"],
        "prescription_management": ["list"],
        "otc_medications": ["list"],
        "durable_equipment": ["list"],
        "social_determinants": ["list"],
        "diagnostic_procedures": ["list"],
        "risk_factors": ["list"],
        "risk_level": "minimal/low/moderate/high"
    },
    "extraction_notes": "detailed explanation of findings"
}
"""

    @staticmethod
    def get_cross_validation_prompt() -> str:
        """Prompt for cross-validating results across tables"""
        return """
You are an expert medical coder performing cross-validation of MDM table results.

TASK: Cross-validate the results from all three MDM tables for consistency and accuracy.

CROSS-VALIDATION CRITERIA:
1. Check for logical consistency between table results
2. Verify that the extracted information supports the assigned levels
3. Identify any contradictions or inconsistencies
4. Ensure the overall clinical picture supports the MDM level
5. Validate that the reasoning is clinically sound

TABLE RESULTS TO VALIDATE:
Problems: {problems_result}
Data: {data_result}
Risk: {risk_result}

PATIENT RECORD CONTEXT:
{patient_record}

PERFORM CROSS-VALIDATION AND PROVIDE:
1. Are the results consistent? (Yes/No)
2. Any contradictions identified
3. Clinical reasoning validation
4. Confidence in cross-validation
5. Final validated results

Respond in JSON format:
{
    "is_consistent": true,
    "contradictions": ["list of any contradictions"],
    "clinical_validation": "detailed clinical reasoning",
    "cross_validation_confidence": 0.95,
    "validated_results": {
        "problems": "validated level and reasoning",
        "data": "validated level and reasoning",
        "risk": "validated level and reasoning"
    },
    "overall_assessment": "comprehensive assessment of consistency"
}
""" 


data_review = '''Here are the revised definitions for each MDM data extraction element, updated based on the AMA 2023 E/M Guidelines PDF you provided:

⸻

✅ Updated Definitions for MDM Data Elements (per 2023 E/M Guidelines)

1. Tests Ordered During Encounter

Updated Definition:
Tests are defined as imaging, laboratory, psychometric, or physiologic data. When ordered during the encounter, they are presumed to be analyzed and count as part of MDM. A clinical panel (e.g., BMP) is counted as a single test, and repeated results of the same test are counted only once per encounter ￼.

⸻

2. Test Results Reviewed

Updated Definition:
Reviewing the result(s) of any unique test (from any source) during the encounter counts as a data point. Multiple results of the same test (e.g., serial glucose levels) count as one unique test. Review must be documented during the encounter to count ￼.

⸻

3. External Records Reviewed

Updated Definition:
Includes review of records, communications, and/or test results from external physicians, facilities, or healthcare organizations. An external source is defined as outside the reporting provider’s group and specialty/subspecialty ￼.

⸻

4. Independent Historian

Updated Definition:
An individual (e.g., parent, guardian, surrogate, witness) who provides history in addition to or instead of the patient due to patient incapacity (e.g., dementia, developmental delay, language barrier not included). More than one source may count if needed due to conflicting or unreliable histories ￼.

⸻

5. Independent Interpretations

Updated Definition:
The provider personally interprets a test (e.g., EKG, CXR) that has a CPT code and where an interpretation or report is customary. The provider cannot also report the professional component (e.g., cannot bill separately for it). A brief notation suffices, but it must represent personal analysis ￼.

⸻

6. External Discussions

Updated Definition:
Includes direct (not staff-mediated) interactive communication with an external physician, QHP, or appropriate source about management or test interpretation. May be asynchronous (e.g., message exchange), but must occur within a short period and directly impact decision-making. Notes alone or chart-sharing without dialogue do not count ￼.

⸻

'''
