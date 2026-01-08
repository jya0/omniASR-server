
import re

class OutputFilter:
    """Filters hallucinations and nonsensical output. (Copied logic for standalone testing)"""
    
    BLOCKLIST = [
        "Amara.org",
        "Subtitles by",
        "Translated by",
        "Copyright",
        "All rights reserved",
        "Paradox Interactive",
        "Mojang AB",
        "Step Id:",
        # "Master Scribes", # Removed from test copy if needed, but keeping for parity
        "Miraculous Ladybug",
        "The following content has been modified",
        "monsieur lucky",
        "marie mathew",
        "thiệt hơn chị bản",
        "dem crite",
        "électrique",
    ]

    @staticmethod
    def is_valid_text(text: str) -> bool:
        if not text or len(text.strip()) < 2:
            return True 
            
        lower_text = text.lower()
        for phrase in OutputFilter.BLOCKLIST:
            if phrase.lower() in lower_text:
                return False

        valid_pattern = re.compile(r'[\u0600-\u06FFa-zA-Z0-9\s.,?!\'"\-:;()]')
        valid_chars = len(valid_pattern.findall(text))
        total_chars = len(text)
        
        if total_chars == 0: return True
        ratio = valid_chars / total_chars
        
        return ratio >= 0.8

def test_filter():
    test_cases = [
        ("Hello world", True),
        ("السلام عليكم", True),
        ("Subtitles by Amara.org", False),
        ("Monsieur Lucky", False),
        ("một số điện được", False), # Vietnamese
        ("This is a test with valid mixed text.", True),
        ("Hello 123!", True),
        ("안녕하세요", False), # Korean (0% valid)
        ("My name is Yao.", True),
    ]
    
    failed = False
    for text, expected in test_cases:
        result = OutputFilter.is_valid_text(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] Text: '{text}' -> Got: {result}, Expected: {expected}")
        if result != expected:
            failed = True
            
    if failed:
        print("Tests FAILED")
        exit(1)
    else:
        print("All Tests PASSED")

if __name__ == "__main__":
    test_filter()
