CRITICAL_ATTACKS = {"Worms", "Backdoor"}
HIGH_ATTACKS     = {"DoS", "Exploits", "Shellcode"}
MEDIUM_ATTACKS   = {"Fuzzers", "Analysis"}

def compute_severity(prediction: str, attack_type: str, alert_count: int) -> str:
    """
    Computes a sophisticated severity score based primarily on the 
    attack category threat level, with reputation modifiers.
    """
    if prediction == "Normal":
        return "Normal"
    
    # 1. Start with the baseline for the attack type
    if attack_type in CRITICAL_ATTACKS:
        severity_score = 4 # Critical
    elif attack_type in HIGH_ATTACKS:
        severity_score = 3 # High
    elif attack_type in MEDIUM_ATTACKS:
        severity_score = 2 # Medium
    else:
        severity_score = 1 # Low (e.g. Generic, Reconnaissance)

    # 2. Add Reputation Modifier (Repeat Offenders)
    # If an IP has attacked more than 5 times, bump severity by 1
    if alert_count >= 5:
        severity_score += 1
    
    # 3. Map back to labels
    if severity_score >= 4: return "Critical"
    if severity_score == 3: return "High"
    if severity_score == 2: return "Medium"
    return "Low"
