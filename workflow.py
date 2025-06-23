"""
EM MDM Extraction Workflow
Implements the complete workflow for extracting MDM levels from patient medical records
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import openai
from openai import OpenAI
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MDMLevel(Enum):
    """MDM Level enumeration"""
    STRAIGHTFORWARD = "Straightforward"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"

@dataclass
class WorkflowStep:
    """Represents a step in the workflow"""
    step_name: str
    input_data: Dict
    output_data: Dict
    confidence_score: float
    execution_time: float
    errors: List[str]

@dataclass
class WorkflowResult:
    """Complete workflow result"""
    patient_id: str
    steps: List[WorkflowStep]
    final_mdm_level: MDMLevel
    overall_confidence: float
    validation_passed: bool
    cross_validation_passed: bool
    total_execution_time: float
    summary: Dict

class EMMDMWorkflow:
    """Complete workflow for EM MDM extraction"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Initialize the workflow
        
        Args:
            api_key: OpenAI API key
            model: Model to use for analysis
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> dict:
        """Load all prompts from prompts.json"""
        with open(os.path.join(os.path.dirname(__file__), 'prompts.json'), 'r') as f:
            return json.load(f)
    
    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """Make API call to LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from LLM"""
        try:
            # Extract JSON from response if it's wrapped in markdown
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error(f"Response: {response}")
            raise
    
    def step_1_extract_problems(self, patient_record: str) -> WorkflowStep:
        """Step 1: Extract information for Problems table"""
        logger.info("Executing Step 1: Problems Extraction")
        
        import time
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_problems_extraction_prompt().format(
                patient_record=patient_record
            )
            
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            
            execution_time = time.time() - start_time
            
            return WorkflowStep(
                step_name="Problems Extraction",
                input_data={"patient_record": patient_record[:500] + "..."},
                output_data=result,
                confidence_score=0.9,  # Default confidence for extraction
                execution_time=execution_time,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in Problems extraction: {e}")
            
            return WorkflowStep(
                step_name="Problems Extraction",
                input_data={"patient_record": patient_record[:500] + "..."},
                output_data={},
                confidence_score=0.0,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def step_2_match_problems(self, extracted_info: Dict) -> WorkflowStep:
        """Step 2: Match Problems table criteria"""
        logger.info("Executing Step 2: Problems Matching")
        
        import time
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_problems_matching_prompt().format(
                extracted_info=json.dumps(extracted_info, indent=2)
            )
            
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            
            execution_time = time.time() - start_time
            
            return WorkflowStep(
                step_name="Problems Matching",
                input_data=extracted_info,
                output_data=result,
                confidence_score=result.get("confidence_score", 0.8),
                execution_time=execution_time,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in Problems matching: {e}")
            
            return WorkflowStep(
                step_name="Problems Matching",
                input_data=extracted_info,
                output_data={},
                confidence_score=0.0,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def step_3_extract_data(self, patient_record: str) -> WorkflowStep:
        """Step 3: Extract information for Data table"""
        logger.info("Executing Step 3: Data Extraction")
        
        import time
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_data_extraction_prompt().format(
                patient_record=patient_record
            )
            
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            
            execution_time = time.time() - start_time
            
            return WorkflowStep(
                step_name="Data Extraction",
                input_data={"patient_record": patient_record[:500] + "..."},
                output_data=result,
                confidence_score=0.9,
                execution_time=execution_time,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in Data extraction: {e}")
            
            return WorkflowStep(
                step_name="Data Extraction",
                input_data={"patient_record": patient_record[:500] + "..."},
                output_data={},
                confidence_score=0.0,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def step_4_match_data(self, extracted_info: Dict) -> WorkflowStep:
        """Step 4: Match Data table criteria"""
        logger.info("Executing Step 4: Data Matching")
        
        import time
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_data_matching_prompt().format(
                extracted_info=json.dumps(extracted_info, indent=2)
            )
            
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            
            execution_time = time.time() - start_time
            
            return WorkflowStep(
                step_name="Data Matching",
                input_data=extracted_info,
                output_data=result,
                confidence_score=result.get("confidence_score", 0.8),
                execution_time=execution_time,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in Data matching: {e}")
            
            return WorkflowStep(
                step_name="Data Matching",
                input_data=extracted_info,
                output_data={},
                confidence_score=0.0,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def step_5_extract_risk(self, patient_record: str) -> WorkflowStep:
        """Step 5: Extract information for Risk table"""
        logger.info("Executing Step 5: Risk Extraction")
        
        import time
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_risk_extraction_prompt().format(
                patient_record=patient_record
            )
            
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            
            execution_time = time.time() - start_time
            
            return WorkflowStep(
                step_name="Risk Extraction",
                input_data={"patient_record": patient_record[:500] + "..."},
                output_data=result,
                confidence_score=0.9,
                execution_time=execution_time,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in Risk extraction: {e}")
            
            return WorkflowStep(
                step_name="Risk Extraction",
                input_data={"patient_record": patient_record[:500] + "..."},
                output_data={},
                confidence_score=0.0,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def step_6_match_risk(self, extracted_info: Dict) -> WorkflowStep:
        """Step 6: Match Risk table criteria"""
        logger.info("Executing Step 6: Risk Matching")
        
        import time
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_risk_matching_prompt().format(
                extracted_info=json.dumps(extracted_info, indent=2)
            )
            
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            
            execution_time = time.time() - start_time
            
            return WorkflowStep(
                step_name="Risk Matching",
                input_data=extracted_info,
                output_data=result,
                confidence_score=result.get("confidence_score", 0.8),
                execution_time=execution_time,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in Risk matching: {e}")
            
            return WorkflowStep(
                step_name="Risk Matching",
                input_data=extracted_info,
                output_data={},
                confidence_score=0.0,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def step_7_final_evaluation(self, problems_result: Dict, data_result: Dict, risk_result: Dict) -> WorkflowStep:
        """Step 7: Final MDM evaluation"""
        logger.info("Executing Step 7: Final Evaluation")
        
        import time
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_final_evaluation_prompt().format(
                problems_level=problems_result.get("predicted_level", "Unknown"),
                problems_confidence=problems_result.get("confidence_score", 0),
                data_level=data_result.get("predicted_level", "Unknown"),
                data_confidence=data_result.get("confidence_score", 0),
                risk_level=risk_result.get("predicted_level", "Unknown"),
                risk_confidence=risk_result.get("confidence_score", 0),
                problems_details=problems_result.get("reasoning", ""),
                data_details=data_result.get("reasoning", ""),
                risk_details=risk_result.get("reasoning", "")
            )
            
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            
            execution_time = time.time() - start_time
            
            return WorkflowStep(
                step_name="Final Evaluation",
                input_data={
                    "problems": problems_result,
                    "data": data_result,
                    "risk": risk_result
                },
                output_data=result,
                confidence_score=result.get("confidence_score", 0.8),
                execution_time=execution_time,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in Final evaluation: {e}")
            
            return WorkflowStep(
                step_name="Final Evaluation",
                input_data={
                    "problems": problems_result,
                    "data": data_result,
                    "risk": risk_result
                },
                output_data={},
                confidence_score=0.0,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def step_8_validation(self, final_result: Dict, patient_record: str) -> WorkflowStep:
        """Step 8: Validation of final result"""
        logger.info("Executing Step 8: Validation")
        
        import time
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_validation_prompt().format(
                mdm_result=json.dumps(final_result, indent=2),
                patient_summary=patient_record[:1000] + "..." if len(patient_record) > 1000 else patient_record
            )
            
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            
            execution_time = time.time() - start_time
            
            return WorkflowStep(
                step_name="Validation",
                input_data=final_result,
                output_data=result,
                confidence_score=result.get("validation_confidence", 0.8),
                execution_time=execution_time,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in Validation: {e}")
            
            return WorkflowStep(
                step_name="Validation",
                input_data=final_result,
                output_data={},
                confidence_score=0.0,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def step_9_cross_validation(self, all_results: Dict, patient_record: str) -> WorkflowStep:
        """Step 9: Cross-validation across all tables"""
        logger.info("Executing Step 9: Cross-Validation")
        
        import time
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_cross_validation_prompt().format(
                problems_result=json.dumps(all_results.get("problems", {}), indent=2),
                data_result=json.dumps(all_results.get("data", {}), indent=2),
                risk_result=json.dumps(all_results.get("risk", {}), indent=2),
                patient_record=patient_record[:1000] + "..." if len(patient_record) > 1000 else patient_record
            )
            
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            
            execution_time = time.time() - start_time
            
            return WorkflowStep(
                step_name="Cross-Validation",
                input_data=all_results,
                output_data=result,
                confidence_score=result.get("cross_validation_confidence", 0.8),
                execution_time=execution_time,
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in Cross-validation: {e}")
            
            return WorkflowStep(
                step_name="Cross-Validation",
                input_data=all_results,
                output_data={},
                confidence_score=0.0,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def execute_workflow(self, patient_record: str, patient_id: str = "unknown") -> WorkflowResult:
        """
        Execute the complete EM MDM extraction workflow
        
        Args:
            patient_record: Complete patient medical record text
            patient_id: Patient identifier
            
        Returns:
            WorkflowResult with complete analysis
        """
        logger.info(f"Starting EM MDM workflow for patient: {patient_id}")
        
        import time
        total_start_time = time.time()
        
        steps = []
        
        try:
            # Step 1: Extract Problems information
            step1 = self.step_1_extract_problems(patient_record)
            steps.append(step1)
            
            # Step 2: Match Problems criteria
            step2 = self.step_2_match_problems(step1.output_data)
            steps.append(step2)
            
            # Step 3: Extract Data information
            step3 = self.step_3_extract_data(patient_record)
            steps.append(step3)
            
            # Step 4: Match Data criteria
            step4 = self.step_4_match_data(step3.output_data)
            steps.append(step4)
            
            # Step 5: Extract Risk information
            step5 = self.step_5_extract_risk(patient_record)
            steps.append(step5)
            
            # Step 6: Match Risk criteria
            step6 = self.step_6_match_risk(step5.output_data)
            steps.append(step6)
            
            # Step 7: Final evaluation
            step7 = self.step_7_final_evaluation(
                step2.output_data,
                step4.output_data,
                step6.output_data
            )
            steps.append(step7)
            
            # Step 8: Validation
            step8 = self.step_8_validation(step7.output_data, patient_record)
            steps.append(step8)
            
            # Step 9: Cross-validation
            all_results = {
                "problems": step2.output_data,
                "data": step4.output_data,
                "risk": step6.output_data,
                "final": step7.output_data
            }
            step9 = self.step_9_cross_validation(all_results, patient_record)
            steps.append(step9)
            
            # Calculate final results
            total_execution_time = time.time() - total_start_time
            
            # Determine final MDM level
            final_level = MDMLevel(step8.output_data.get("final_validated_level", "Low"))
            
            # Calculate overall confidence
            valid_steps = [s for s in steps if s.confidence_score > 0]
            overall_confidence = sum(s.confidence_score for s in valid_steps) / len(valid_steps) if valid_steps else 0
            
            # Check validation status
            validation_passed = step8.output_data.get("is_valid", False)
            cross_validation_passed = step9.output_data.get("is_consistent", False)
            
            # Create summary
            summary = {
                "problems_level": step2.output_data.get("predicted_level", "Unknown"),
                "data_level": step4.output_data.get("predicted_level", "Unknown"),
                "risk_level": step6.output_data.get("predicted_level", "Unknown"),
                "final_level": final_level.value,
                "validation_issues": step8.output_data.get("issues_identified", []),
                "cross_validation_contradictions": step9.output_data.get("contradictions", []),
                "recommendations": step8.output_data.get("recommendations", [])
            }
            
            result = WorkflowResult(
                patient_id=patient_id,
                steps=steps,
                final_mdm_level=final_level,
                overall_confidence=overall_confidence,
                validation_passed=validation_passed,
                cross_validation_passed=cross_validation_passed,
                total_execution_time=total_execution_time,
                summary=summary
            )
            
            logger.info(f"Workflow completed for patient {patient_id}. Final MDM level: {final_level.value}")
            return result
            
        except Exception as e:
            total_execution_time = time.time() - total_start_time
            logger.error(f"Error in workflow execution: {e}")
            
            # Return partial result
            return WorkflowResult(
                patient_id=patient_id,
                steps=steps,
                final_mdm_level=MDMLevel.LOW,  # Default fallback
                overall_confidence=0.0,
                validation_passed=False,
                cross_validation_passed=False,
                total_execution_time=total_execution_time,
                summary={"error": str(e)}
            )
    
    def generate_workflow_report(self, result: WorkflowResult) -> str:
        """Generate a detailed workflow report"""
        report = f"""
=== EM MDM EXTRACTION WORKFLOW REPORT ===
Patient ID: {result.patient_id}
Final MDM Level: {result.final_mdm_level.value}
Overall Confidence: {result.overall_confidence:.2f}
Total Execution Time: {result.total_execution_time:.2f} seconds
Validation Passed: {result.validation_passed}
Cross-Validation Passed: {result.cross_validation_passed}

=== STEP-BY-STEP ANALYSIS ===
"""
        
        for i, step in enumerate(result.steps, 1):
            report += f"""
Step {i}: {step.step_name}
- Execution Time: {step.execution_time:.2f} seconds
- Confidence Score: {step.confidence_score:.2f}
- Errors: {len(step.errors)}
"""
            if step.errors:
                report += f"- Error Details: {', '.join(step.errors)}\n"
        
        report += f"""
=== SUMMARY ===
Problems Level: {result.summary.get('problems_level', 'Unknown')}
Data Level: {result.summary.get('data_level', 'Unknown')}
Risk Level: {result.summary.get('risk_level', 'Unknown')}
Final Level: {result.summary.get('final_level', 'Unknown')}

Validation Issues: {result.summary.get('validation_issues', [])}
Cross-Validation Contradictions: {result.summary.get('cross_validation_contradictions', [])}
Recommendations: {result.summary.get('recommendations', [])}
"""
        
        return report

def main():
    """Example usage of the workflow"""
    
    # Example patient record
    sample_patient_record = """
    CHIEF COMPLAINT: Chest pain and shortness of breath
    
    HPI: 65-year-old male presents with 3-day history of chest pain and shortness of breath. 
    Pain is substernal, radiates to left arm, worse with exertion. Associated with diaphoresis and nausea.
    Patient has history of hypertension, diabetes, and previous MI 2 years ago.
    
    ASSESSMENT:
    1. Acute coronary syndrome - rule out MI
    2. Hypertension - uncontrolled
    3. Diabetes mellitus type 2 - stable
    4. Anxiety
    
    PLAN:
    - ECG: ST elevation in anterior leads
    - Troponin I: 2.5 ng/mL (elevated)
    - CBC: WBC 12,000, Hgb 14.2
    - CMP: Glucose 180, BUN 25, Creatinine 1.2
    - Chest X-ray: Normal cardiac silhouette
    - Cardiology consultation ordered
    - Aspirin 325mg given
    - Nitroglycerin sublingual PRN
    - Metoprolol 25mg BID
    - Lisinopril 10mg daily
    - Metformin 500mg BID continued
    
    LAB ORDERS:
    - Repeat troponin in 6 hours
    - Lipid panel
    - HbA1c
    
    MEDICATIONS:
    - Aspirin 325mg (new)
    - Nitroglycerin sublingual PRN (new)
    - Metoprolol 25mg BID (new)
    - Lisinopril 10mg daily (continued)
    - Metformin 500mg BID (continued)
    
    RISK FACTORS:
    - Previous MI
    - Hypertension
    - Diabetes
    - Age >65
    - Male gender
    """
    
    try:
        # Initialize workflow
        workflow = EMMDMWorkflow()
        
        # Execute workflow
        result = workflow.execute_workflow(sample_patient_record, "PATIENT_001")
        
        # Generate report
        report = workflow.generate_workflow_report(result)
        print(report)
        
        # Save detailed results
        with open("workflow_results.json", "w") as f:
            json.dump({
                "patient_id": result.patient_id,
                "final_mdm_level": result.final_mdm_level.value,
                "overall_confidence": result.overall_confidence,
                "validation_passed": result.validation_passed,
                "cross_validation_passed": result.cross_validation_passed,
                "total_execution_time": result.total_execution_time,
                "summary": result.summary,
                "steps": [
                    {
                        "step_name": step.step_name,
                        "confidence_score": step.confidence_score,
                        "execution_time": step.execution_time,
                        "errors": step.errors,
                        "output_data": step.output_data
                    }
                    for step in result.steps
                ]
            }, f, indent=2)
        
        print(f"\nDetailed results saved to workflow_results.json")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 