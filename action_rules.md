# Bot Action Rules & Restrictions

This document outlines the **14-Action Space** available to the AI agent and the strict rules governing its behavior.

## Action Space (Discrete 14)

| Index | Action Name | Description | Restrictions / Notes |
| :--- | :--- | :--- | :--- |
| **0** | **Fold** | Fold hand | Cannot fold if Check is available (Open-Folding banned). |
| **1** | **Check/Call** | Check if no bet, Call if facing bet | |
| **2** | **Min-Raise Only** | Raise to minimum allowed amount | **BANNED in Open Pot** (Cannot Min-Bet). Only available when facing a bet. |
| **3** | **Bet 10%** | Bet/Raise 10% of Pot | **Strict Masking:** Disabled if amount < 1 BB (Min-Bet). |
| **4** | **Bet 25%** | Bet/Raise 25% of Pot | **Strict Masking:** Disabled if amount < 1 BB. |
| **5** | **Bet 33%** | Bet/Raise 33% of Pot | **Strict Masking:** Disabled if amount < 1 BB. |
| **6** | **Bet 50%** | Bet/Raise 50% of Pot | **Strict Masking:** Disabled if amount < 1 BB. |
| **7** | **Bet 75%** | Bet/Raise 75% of Pot | **Strict Masking:** Disabled if amount < 1 BB. |
| **8** | **Bet 100%** | Bet/Raise 100% of Pot (Pot Bet) | **Strict Masking:** Disabled if amount < 1 BB. |
| **9** | **Bet 125%** | Bet/Raise 125% of Pot | **Strict Masking:** Disabled if amount < 1 BB. |
| **10** | **Bet 150%** | Bet/Raise 150% of Pot | **Strict Masking:** Disabled if amount < 1 BB. |
| **11** | **Bet 200%** | Bet/Raise 200% of Pot | **Strict Masking:** Disabled if amount < 1 BB. |
| **12** | **Bet 300%** | Bet/Raise 300% of Pot | **Strict Masking:** Disabled if amount < 1 BB. |
| **13** | **All-In** | Bet/Raise all remaining chips | Always available. |

## Key Rules & Laws

### 1. The "No Min-Bet" Law
*   **Rule:** The AI is **strictly prohibited** from making a Minimum Bet (1 BB) into an open pot.
*   **Implementation:**
    *   **Action 2 (Min-Raise)** is **MASKED** (disabled) when `current_bet == 0`.
    *   **Safety Net:** If the AI somehow selects Action 2 in an open pot, the environment forces a **Check** action instead of a Min-Bet.

### 2. The "Strict Masking" Law (No Auto-Correction)
*   **Rule:** The AI cannot select a percentage bet if the calculated amount is less than the legal minimum bet (1 BB).
*   **Implementation:**
    *   If `(Pot * Percentage) < 1 BB`, the corresponding action is **MASKED**.
    *   **Example:** In a 1.5 BB pot, 10% (0.15 BB) and 50% (0.75 BB) actions are disabled. The AI must choose 75% (1.125 BB) or higher.
    *   **Reason:** Prevents "Intent-Execution Mismatch" where the AI thinks it's betting small (10%) but actually bets large (Min-Bet = 66% of pot).

### 3. The "Min-Raise" Exception
*   **Rule:** While Min-Bet is banned, **Min-Raise** is allowed and encouraged as a strategic option.
*   **Implementation:**
    *   Action 2 is **ENABLED** when `current_bet > 0` (facing a bet).
    *   This allows the AI to make the smallest legal raise against an opponent's bet.

### 4. Short Stack Logic
*   **Rule:** If a player has fewer chips than the required minimum bet/raise, they can only **All-In**.
*   **Implementation:**
    *   If `Chips < Min_Bet`, all Bet/Raise actions are masked except **All-In (Action 13)**.
