"""
EM MDM Extractor - Medical Decision Making Level Extraction
Uses AMA Guidelines for Evaluation and Management Coding

This module provides a comprehensive workflow to extract MDM levels from patient medical records
by analyzing Problems, Data, and Risk tables according to AMA guidelines.
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
class MDMTableResult:
    """Result for each MDM table analysis"""
    table_name: str
    extracted_info: Dict
    matched_criteria: List[str]
    predicted_level: MDMLevel
    confidence_score: float
    reasoning: str

@dataclass
class MDMFinalResult:
    """Final MDM result combining all tables"""
    problems_level: MDMLevel
    data_level: MDMLevel
    risk_level: MDMLevel
    final_level: MDMLevel
    reasoning: str
    confidence_score: float

class EMMDMExtractor:
    """Main class for EM MDM extraction"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Initialize the EM MDM Extractor
        
        Args:
            api_key: OpenAI API key
            model: Model to use for analysis
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # Load prompts
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
    
    def extract_problems_info(self, patient_record: str) -> Dict:
        """Extract information relevant to Problems table"""
        logger.info("Extracting Problems table information...")
        
        prompt = self.prompts["problems_extraction"].format(
            patient_record=patient_record
        )
        
        response = self._call_llm(prompt)
        return self._parse_json_response(response)
    
    def match_problems_criteria(self, extracted_info: Dict) -> Dict:
        """Match extracted information to Problems table criteria"""
        logger.info("Matching Problems table criteria...")
        
        prompt = self.prompts["problems_matching"].format(
            extracted_info=json.dumps(extracted_info, indent=2)
        )
        
        response = self._call_llm(prompt)
        return self._parse_json_response(response)
    
    def extract_data_info(self, patient_record: str) -> Dict:
        """Extract information relevant to Data table"""
        logger.info("Extracting Data table information...")
        
        prompt = self.prompts["data_extraction"].format(
            patient_record=patient_record
        )
        
        response = self._call_llm(prompt)
        return self._parse_json_response(response)
    
    def match_data_criteria(self, extracted_info: Dict) -> Dict:
        """Match extracted information to Data table criteria"""
        logger.info("Matching Data table criteria...")
        
        prompt = self.prompts["data_matching"].format(
            extracted_info=json.dumps(extracted_info, indent=2)
        )
        
        response = self._call_llm(prompt)
        return self._parse_json_response(response)
    
    def extract_risk_info(self, patient_record: str) -> Dict:
        """Extract information relevant to Risk table"""
        logger.info("Extracting Risk table information...")
        
        prompt = self.prompts["risk_extraction"].format(
            patient_record=patient_record
        )
        
        response = self._call_llm(prompt)
        return self._parse_json_response(response)
    
    def match_risk_criteria(self, extracted_info: Dict) -> Dict:
        """Match extracted information to Risk table criteria"""
        logger.info("Matching Risk table criteria...")
        
        prompt = self.prompts["risk_matching"].format(
            extracted_info=json.dumps(extracted_info, indent=2)
        )
        
        response = self._call_llm(prompt)
        return self._parse_json_response(response)
    
    def evaluate_final_mdm(self, problems_result: Dict, data_result: Dict, risk_result: Dict) -> Dict:
        """Evaluate final MDM level based on all table results"""
        logger.info("Evaluating final MDM level...")
        
        prompt = self.prompts["final_evaluation"].format(
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
        return self._parse_json_response(response)
    
    def validate_result(self, mdm_result: Dict, patient_record: str) -> Dict:
        """Validate the final MDM result"""
        logger.info("Validating MDM result...")
        
        prompt = self.prompts["validation"].format(
            mdm_result=json.dumps(mdm_result, indent=2),
            patient_summary=patient_record[:1000] + "..." if len(patient_record) > 1000 else patient_record
        )
        
        response = self._call_llm(prompt)
        return self._parse_json_response(response)
    
    def process_patient_record(self, patient_record: str) -> MDMFinalResult:
        """
        Main workflow to process a patient record and extract MDM level
        
        Args:
            patient_record: Complete patient medical record text
            
        Returns:
            MDMFinalResult with all analysis results
        """
        logger.info("Starting MDM extraction workflow...")
        
        try:
            # Step 1: Extract information for each table
            problems_extracted = self.extract_problems_info(patient_record)
            data_extracted = self.extract_data_info(patient_record)
            risk_extracted = self.extract_risk_info(patient_record)
            
            # Step 2: Match criteria for each table
            problems_matched = self.match_problems_criteria(problems_extracted)
            data_matched = self.match_data_criteria(data_extracted)
            risk_matched = self.match_risk_criteria(risk_extracted)
            
            # Step 3: Evaluate final MDM level
            final_evaluation = self.evaluate_final_mdm(problems_matched, data_matched, risk_matched)
            
            # Step 4: Validate result
            validation = self.validate_result(final_evaluation, patient_record)
            
            # Create final result
            result = MDMFinalResult(
                problems_level=MDMLevel(problems_matched["predicted_level"]),
                data_level=MDMLevel(data_matched["predicted_level"]),
                risk_level=MDMLevel(risk_matched["predicted_level"]),
                final_level=MDMLevel(validation["final_validated_level"]),
                reasoning=validation["validation_reasoning"],
                confidence_score=validation["validation_confidence"]
            )
            
            logger.info(f"MDM extraction completed. Final level: {result.final_level.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error in MDM extraction workflow: {e}")
            raise
    
    def get_detailed_analysis(self, patient_record: str) -> Dict:
        """
        Get detailed analysis with all intermediate results
        
        Args:
            patient_record: Complete patient medical record text
            
        Returns:
            Dictionary with all analysis steps and results
        """
        logger.info("Performing detailed MDM analysis...")
        
        try:
            # Step 1: Extract information for each table
            problems_extracted = self.extract_problems_info(patient_record)
            data_extracted = self.extract_data_info(patient_record)
            risk_extracted = self.extract_risk_info(patient_record)
            
            # Step 2: Match criteria for each table
            problems_matched = self.match_problems_criteria(problems_extracted)
            data_matched = self.match_data_criteria(data_extracted)
            risk_matched = self.match_risk_criteria(risk_extracted)
            
            # Step 3: Evaluate final MDM level
            final_evaluation = self.evaluate_final_mdm(problems_matched, data_matched, risk_matched)
            
            # Step 4: Validate result
            validation = self.validate_result(final_evaluation, patient_record)
            
            return {
                "extraction_results": {
                    "problems": problems_extracted,
                    "data": data_extracted,
                    "risk": risk_extracted
                },
                "matching_results": {
                    "problems": problems_matched,
                    "data": data_matched,
                    "risk": risk_matched
                },
                "final_evaluation": final_evaluation,
                "validation": validation,
                "summary": {
                    "problems_level": problems_matched["predicted_level"],
                    "data_level": data_matched["predicted_level"],
                    "risk_level": risk_matched["predicted_level"],
                    "final_level": validation["final_validated_level"],
                    "overall_confidence": validation["validation_confidence"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in detailed analysis: {e}")
            raise

def main():
    """Example usage of the EM MDM Extractor"""
    
    # Example patient record (replace with actual patient data)
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
        # Initialize extractor
        extractor = EMMDMExtractor()
        
        # Process patient record
        result = extractor.process_patient_record(sample_patient_record)
        
        print("=== MDM EXTRACTION RESULTS ===")
        print(f"Problems Level: {result.problems_level.value}")
        print(f"Data Level: {result.data_level.value}")
        print(f"Risk Level: {result.risk_level.value}")
        print(f"Final MDM Level: {result.final_level.value}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Reasoning: {result.reasoning}")
        
        # Get detailed analysis
        detailed = extractor.get_detailed_analysis(sample_patient_record)
        
        print("\n=== DETAILED ANALYSIS ===")
        print(json.dumps(detailed, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 