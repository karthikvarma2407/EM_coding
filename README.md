# EM MDM Extraction System

A comprehensive system for extracting Medical Decision Making (MDM) levels from patient medical records using the 2023 AMA Guidelines for Evaluation and Management (E/M) coding.

## Overview

This system uses advanced LLM (Large Language Model) technology to analyze patient medical records and determine the appropriate MDM level based on three key tables:
- **Problems Table**: Number of diagnoses or management options
- **Data Table**: Amount and/or complexity of data to be reviewed and analyzed
- **Risk Table**: Risk of complications and/or morbidity or mortality

## Features

- **Comprehensive Analysis**: Extracts information from all three MDM tables
- **High Accuracy**: Uses specialized prompts and validation steps
- **Validation System**: Multiple validation layers including cross-validation
- **Detailed Reporting**: Provides step-by-step analysis and reasoning
- **Batch Processing**: Can process multiple patient records efficiently
- **Flexible Workflow**: Supports both simple and complex medical cases

## Project Structure

```
EM_coding/
├── em_mdm_extractor.py    # Main extraction module
├── workflow.py            # Complete workflow implementation
├── prompts.py             # All prompts used in the system
├── example_usage.py       # Example usage demonstrations
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Basic Usage

```python
from em_mdm_extractor import EMMDMExtractor

# Initialize extractor
extractor = EMMDMExtractor()

# Process patient record
patient_record = """
CHIEF COMPLAINT: Chest pain and shortness of breath

HPI: 65-year-old male presents with 3-day history of chest pain...
ASSESSMENT: Acute coronary syndrome, Hypertension, Diabetes
PLAN: ECG, Troponin, Cardiology consultation...
"""

result = extractor.process_patient_record(patient_record)

print(f"Final MDM Level: {result.final_level.value}")
print(f"Confidence: {result.confidence_score:.2f}")
```

### Complete Workflow

```python
from workflow import EMMDMWorkflow

# Initialize workflow
workflow = EMMDMWorkflow()

# Execute complete workflow with validation
result = workflow.execute_workflow(patient_record, "PATIENT_001")

# Generate detailed report
report = workflow.generate_workflow_report(result)
print(report)
```

### Batch Processing

```python
from em_mdm_extractor import EMMDMExtractor

extractor = EMMDMExtractor()

patient_records = [
    {"id": "PATIENT_001", "record": "..."},
    {"id": "PATIENT_002", "record": "..."},
    # ... more records
]

results = []
for patient in patient_records:
    result = extractor.process_patient_record(patient['record'])
    results.append({
        "patient_id": patient['id'],
        "final_level": result.final_level.value,
        "confidence": result.confidence_score
    })
```

## Workflow Steps

The system follows a comprehensive 9-step workflow:

### Step 1: Problems Extraction
- Extracts information relevant to the Problems table
- Identifies number and types of problems
- Assesses severity and complexity

### Step 2: Problems Matching
- Matches extracted information to Problems table criteria
- Determines Problems MDM level
- Provides confidence score and reasoning

### Step 3: Data Extraction
- Extracts information relevant to the Data table
- Identifies lab tests, imaging, documents reviewed
- Assesses data complexity

### Step 4: Data Matching
- Matches extracted information to Data table criteria
- Determines Data MDM level
- Provides confidence score and reasoning

### Step 5: Risk Extraction
- Extracts information relevant to the Risk table
- Identifies surgical procedures, medications, risk factors
- Assesses risk level

### Step 6: Risk Matching
- Matches extracted information to Risk table criteria
- Determines Risk MDM level
- Provides confidence score and reasoning

### Step 7: Final Evaluation
- Combines results from all three tables
- Applies MDM level determination rules
- Determines final MDM level

### Step 8: Validation
- Validates the final result for accuracy
- Checks consistency with AMA guidelines
- Identifies potential issues

### Step 9: Cross-Validation
- Cross-validates results across all tables
- Ensures clinical consistency
- Provides final confidence assessment

## MDM Level Determination

The system determines MDM levels based on the 2023 AMA guidelines:

### Problems Table
- **Minimal**: 1 problem
- **Low**: 2 problems OR 1 stable chronic illness OR 1 acute uncomplicated illness/injury
- **Moderate**: 3 problems OR 1 chronic illness with exacerbation/progression OR 1 acute complicated injury
- **High**: 4+ problems OR 1 chronic illness with severe exacerbation/progression OR 1 acute/chronic illness posing threat to life/body function

### Data Table
- **Minimal**: No data reviewed
- **Low**: 1 test/lab OR 1 document review OR 1 independent historian
- **Moderate**: 2 tests/labs OR 1 test + 1 document OR 1 test + 1 independent historian OR 2 documents OR 2 independent historians
- **High**: 3+ tests/labs OR 2+ tests + 1 document OR 1 test + 2+ documents OR 3+ documents OR 3+ independent historians

### Risk Table
- **Minimal**: No risk
- **Low**: Prescription drug management OR over-the-counter drugs OR durable medical equipment OR social determinants of health
- **Moderate**: Minor surgery OR prescription drug management with monitoring OR social determinants of health with counseling OR diagnostic endoscopies
- **High**: Major surgery OR prescription drug management with intensive monitoring OR diagnostic endoscopies with identified risk factors OR social determinants of health with community resources

### Final Level Rules
- For 2 out of 3 elements meeting or exceeding a level, the overall MDM level is that level
- If 2 out of 3 elements are Moderate, the overall level is Moderate
- If 2 out of 3 elements are High, the overall level is High
- If 2 out of 3 elements are Low, the overall level is Low
- If 2 out of 3 elements are Minimal, the overall level is Minimal

## Validation Features

### Primary Validation
- Checks adherence to AMA guidelines
- Verifies consistency across tables
- Validates reasoning and confidence scores

### Cross-Validation
- Ensures logical consistency between table results
- Identifies contradictions or inconsistencies
- Validates clinical reasoning

### Confidence Scoring
- Each step provides a confidence score (0-1)
- Overall confidence calculated from all steps
- Low confidence triggers additional review

## Example Output

```json
{
  "patient_id": "PATIENT_001",
  "final_mdm_level": "Moderate",
  "overall_confidence": 0.92,
  "validation_passed": true,
  "cross_validation_passed": true,
  "summary": {
    "problems_level": "Moderate",
    "data_level": "Moderate", 
    "risk_level": "Low",
    "final_level": "Moderate",
    "validation_issues": [],
    "recommendations": []
  }
}
```

## Running Examples

Run the example usage script to see the system in action:

```bash
python example_usage.py
```

This will demonstrate:
1. Basic usage with simple cases
2. Complex medical scenarios
3. Batch processing
4. Detailed analysis
5. Validation focus

## Configuration

### Model Selection
You can specify different LLM models:
```python
extractor = EMMDMExtractor(model="gpt-4-turbo")
workflow = EMMDMWorkflow(model="gpt-4")
```

### Temperature Settings
Adjust temperature for different levels of creativity vs consistency:
```python
# More conservative (default)
extractor = EMMDMExtractor()
extractor._call_llm(prompt, temperature=0.1)

# More creative (for edge cases)
extractor._call_llm(prompt, temperature=0.3)
```

## Best Practices

1. **Patient Record Format**: Ensure patient records are well-structured with clear sections (HPI, Assessment, Plan, etc.)

2. **Validation**: Always review validation results and address any issues identified

3. **Confidence Scores**: Pay attention to confidence scores - low confidence may indicate unclear documentation

4. **Cross-Validation**: Use cross-validation for complex cases to ensure accuracy

5. **Documentation**: Keep detailed logs of all analyses for audit purposes

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure OPENAI_API_KEY is set correctly
2. **JSON Parsing Error**: Check if LLM response is properly formatted
3. **Low Confidence**: Review patient record clarity and completeness
4. **Validation Failures**: Check for inconsistencies in extracted information

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **API Costs**: Each analysis uses multiple API calls (9 steps)
- **Processing Time**: Complex cases may take 30-60 seconds
- **Batch Processing**: Use batch processing for multiple records
- **Caching**: Consider caching results for repeated analyses

## Contributing

To improve the system:

1. **Prompt Optimization**: Refine prompts for better accuracy
2. **Validation Rules**: Add additional validation criteria
3. **Edge Cases**: Handle special medical scenarios
4. **Performance**: Optimize API usage and processing time

## License

This project is for educational and research purposes. Please ensure compliance with relevant medical coding regulations and guidelines.

## Support

For questions or issues:
1. Check the example usage files
2. Review the validation results
3. Ensure proper API key configuration
4. Verify patient record format

## Disclaimer

This system is designed to assist with MDM level determination but should not replace professional medical coding expertise. Always verify results against current AMA guidelines and consult with certified medical coders for final determinations. 