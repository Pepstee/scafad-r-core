#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/workspace')

# Execute the production validation
exec(open('/workspace/run_production_validation.py').read())