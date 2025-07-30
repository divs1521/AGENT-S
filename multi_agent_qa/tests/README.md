# Tests Directory

This directory contains all test files for the Multi-Agent QA System, organized by test type.

## Directory Structure

```
tests/
├── __init__.py
├── README.md
├── run_tests.py           # Test runner script
├── api/                   # API-related tests
│   ├── __init__.py
│   ├── test_direct_api.py
│   ├── test_gemini.py
│   ├── test_github_api.py
│   └── debug_gemini.py
├── integration/           # Integration tests
│   ├── __init__.py
│   ├── test_integration.py
│   ├── test_full_execution.py
│   └── setup_integration.py
└── unit/                  # Unit tests
    ├── __init__.py
    └── test_executor_agent.py
```

## Test Types

### Unit Tests (`unit/`)
- Individual component testing
- Isolated functionality verification
- Mock dependencies

### Integration Tests (`integration/`)
- End-to-end workflow testing
- Multi-component interaction
- Agent-S and Android World integration

### API Tests (`api/`)
- LLM provider API testing
- Authentication verification
- Rate limiting and error handling

## Running Tests

### All Tests
```bash
python tests/run_tests.py --type all
```

### Specific Test Types
```bash
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration
python tests/run_tests.py --type api
```

### Individual Test Files
```bash
python tests/unit/test_executor_agent.py
python tests/api/test_github_api.py
```

### Using pytest (if installed)
```bash
pytest tests/
pytest tests/unit/
pytest tests/integration/
```

## Test Environment

Make sure to set up your API keys in `config/api_keys.json` before running tests that require real API calls.

## Writing New Tests

- **Unit tests**: Add to `tests/unit/` for individual component testing
- **Integration tests**: Add to `tests/integration/` for workflow testing  
- **API tests**: Add to `tests/api/` for external service testing

Follow the naming convention `test_*.py` for all test files.
