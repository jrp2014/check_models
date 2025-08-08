#!/usr/bin/env python3

# Test which markdown characters actually break table formatting

def test_markdown_table_safety():
    """Test various markdown characters in table cells."""
    
    print("=== MARKDOWN TABLE CHARACTER SAFETY TEST ===\n")
    
    # Characters that DEFINITELY break tables
    definitely_break = [
        ('|', 'Pipe character - breaks column structure'),
        ('\n', 'Newline - breaks row structure'),
    ]
    
    # Characters that are SAFE in table cells
    safe_chars = [
        ('()', 'Parentheses'),
        ('{}', 'Curly braces'),
        ('<>', 'Angle brackets'),
        ('!', 'Exclamation'),
        ('+', 'Plus'),
        ('-', 'Minus (not at start of line)'),
        ('=', 'Equals'),
        ('~', 'Tilde'),
        ('^', 'Caret'),
        ('&', 'Ampersand'),
        ('%', 'Percent'),
        ('$', 'Dollar'),
        ('@', 'At symbol'),
        ('?', 'Question mark'),
        ('.', 'Period'),
        (',', 'Comma'),
        (';', 'Semicolon'),
        (':', 'Colon'),
        ('"\'', 'Quotes and apostrophes'),
    ]
    
    # Characters with CONDITIONAL effects (context-dependent)
    conditional_chars = [
        ('*', 'Asterisk - creates emphasis, usually safe in table cells'),
        ('_', 'Underscore - creates emphasis, usually safe in table cells'),
        ('#', 'Hash - creates headers, safe in table cells'),
        ('`', 'Backtick - creates code spans, safe in table cells'),
        ('[]', 'Square brackets - part of link syntax, safe alone'),
        ('\\', 'Backslash - escape character, can affect other chars'),
    ]
    
    print("Characters that DEFINITELY BREAK table formatting:")
    for char, desc in definitely_break:
        print(f"  {repr(char):6} - {desc}")
    
    print(f"\nCharacters that are SAFE in table cells ({len(safe_chars)} total):")
    for char, desc in safe_chars:
        print(f"  {char:6} - {desc}")
    
    print(f"\nCharacters with CONDITIONAL effects ({len(conditional_chars)} total):")
    for char, desc in conditional_chars:
        print(f"  {char:6} - {desc}")
    
    print("\n=== ANALYSIS FOR _escape_markdown_in_text() ===")
    print("\nCurrently escaping ALL of these:")
    current_escapes = ['\\', '`', '*', '_', '#', '|', '[', ']']
    for char in current_escapes:
        print(f"  {char}")
    
    print("\nCould avoid escaping these SAFE characters:")
    # None from our current list are definitively safe to unescape
    print("  (None - all current escapes serve a purpose)")
    
    print("\nRecommendation:")
    print("  - Keep escaping | (pipe) - CRITICAL for table structure")
    print("  - Keep escaping \\ (backslash) - affects other characters") 
    print("  - Could consider NOT escaping *, _, #, `, [], ] in some contexts")
    print("  - But better to be safe - current escaping is appropriate")
    
    return current_escapes, safe_chars, conditional_chars

if __name__ == "__main__":
    test_markdown_table_safety()
