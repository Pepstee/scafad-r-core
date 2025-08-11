#!/usr/bin/env python3
import asyncio
from complete_layer0_integration_test import run_complete_integration_test

if __name__ == "__main__":
    result = asyncio.run(run_complete_integration_test())
    print(f"\nFinal Result: {result}")