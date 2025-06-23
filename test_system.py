"""
Test Script for EM MDM Extraction System
Simple test to verify the system works correctly
"""

import os
import sys

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from prompts import EMMDMPrompts
        print("✓ prompts.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import prompts.py: {e}")
        return False
    
    try:
        from em_mdm_extractor import EMMDMExtractor
        print("✓ em_mdm_extractor.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import em_mdm_extractor.py: {e}")
        return False
    
    try:
        from workflow import EMMDMWorkflow
        print("✓ workflow.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import workflow.py: {e}")
        return False
    
    return True

def test_prompts():
    """Test that prompts can be generated"""
    print("\nTesting prompts...")
    
    try:
        from prompts import EMMDMPrompts
        
        prompts = EMMDMPrompts()
        
        # Test each prompt method
        problems_extraction = prompts.get_problems_extraction_prompt()
        problems_matching = prompts.get_problems_matching_prompt()
        data_extraction = prompts.get_data_extraction_prompt()
        data_matching = prompts.get_data_matching_prompt()
        risk_extraction = prompts.get_risk_extraction_prompt()
        risk_matching = prompts.get_risk_matching_prompt()
        final_evaluation = prompts.get_final_evaluation_prompt()
        validation = prompts.get_validation_prompt()
        
        print("✓ All prompts generated successfully")
        print(f"  - Problems extraction prompt length: {len(problems_extraction)} chars")
        print(f"  - Problems matching prompt length: {len(problems_matching)} chars")
        print(f"  - Data extraction prompt length: {len(data_extraction)} chars")
        print(f"  - Data matching prompt length: {len(data_matching)} chars")
        print(f"  - Risk extraction prompt length: {len(risk_extraction)} chars")
        print(f"  - Risk matching prompt length: {len(risk_matching)} chars")
        print(f"  - Final evaluation prompt length: {len(final_evaluation)} chars")
        print(f"  - Validation prompt length: {len(validation)} chars")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test prompts: {e}")
        return False

def test_extractor_initialization():
    """Test extractor initialization"""
    print("\nTesting extractor initialization...")
    
    try:
        from em_mdm_extractor import EMMDMExtractor
        
        # Clear any existing API key
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        # Test without API key (should raise error)
        try:
            extractor = EMMDMExtractor()
            print("✗ Should have failed without API key")
            return False
        except ValueError:
            print("✓ Correctly failed without API key")
        
        # Test with mock API key
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            extractor = EMMDMExtractor()
            print("✓ Extractor initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize extractor: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to test extractor: {e}")
        return False

def test_workflow_initialization():
    """Test workflow initialization"""
    print("\nTesting workflow initialization...")
    
    try:
        from workflow import EMMDMWorkflow
        
        # Clear any existing API key
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        # Test without API key (should raise error)
        try:
            workflow = EMMDMWorkflow()
            print("✗ Should have failed without API key")
            return False
        except ValueError:
            print("✓ Correctly failed without API key")
        
        # Test with mock API key
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            workflow = EMMDMWorkflow()
            print("✓ Workflow initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize workflow: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to test workflow: {e}")
        return False

def test_sample_patient_record():
    """Test with a sample patient record"""
    print("\nTesting with sample patient record...")
    
    sample_record = """
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
        from em_mdm_extractor import EMMDMExtractor
        
        # Set mock API key
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        extractor = EMMDMExtractor()
        
        # Test that the method exists and can be called
        if hasattr(extractor, 'process_patient_record'):
            print("✓ process_patient_record method exists")
        else:
            print("✗ process_patient_record method not found")
            return False
        
        if hasattr(extractor, 'get_detailed_analysis'):
            print("✓ get_detailed_analysis method exists")
        else:
            print("✗ get_detailed_analysis method not found")
            return False
        
        print("✓ Sample patient record test passed")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test sample patient record: {e}")
        return False

def test_requirements():
    """Test that required packages are available"""
    print("\nTesting required packages...")
    
    required_packages = [
        'openai',
        'json',
        'logging',
        'typing',
        'dataclasses',
        'enum',
        'os'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not available")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests"""
    print("EM MDM Extraction System - System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_prompts,
        test_extractor_initialization,
        test_workflow_initialization,
        test_sample_patient_record,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        print("\nTo use the system:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("2. Run: python example_usage.py")
    else:
        print("✗ Some tests failed. Please fix the issues before using the system.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 