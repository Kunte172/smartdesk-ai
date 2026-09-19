from __future__ import annotations

from enum import StrEnum


class TicketCategory(StrEnum):
    ACCOUNT_ACCESS = "ACCOUNT_ACCESS"
    BILLING = "BILLING"
    DELIVERY = "DELIVERY"
    GENERAL = "GENERAL"
    REFUND = "REFUND"
    TECHNICAL = "TECHNICAL"


class TicketPriority(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"
