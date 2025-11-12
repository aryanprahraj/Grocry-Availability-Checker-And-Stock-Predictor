# Grocery Availability Checker and Stock Predictor - CI/CD Implementation

## 🎯 Overview
A complete CI/CD pipeline has been implemented for the Grocery Stock Predictor project using GitHub Actions, Docker, and modern DevOps practices.

## 📁 New Files Created

### GitHub Actions Workflows (`.github/workflows/`)
1. **`ci.yml`** - Continuous Integration Pipeline
   - Multi-platform testing (Ubuntu, Windows, macOS)
   - Python versions: 3.9, 3.10, 3.11
   - Automated testing, linting, and code quality checks
   - Security scanning and dependency vulnerability checks

2. **`cd.yml`** - Continuous Deployment Pipeline
   - Automated model training (manual trigger)
   - Docker image building and pushing
   - Staging and production deployments
   - Environment-based deployment protection

3. **`dependency-updates.yml`** - Automated Dependency Updates
   - Weekly dependency update checks
   - Automatic PR creation for updates
   - Keeps project secure and up-to-date

### Docker Configuration
1. **`Dockerfile`** - Container configuration
   - Python 3.10 slim base image
   - Optimized for ML workloads
   - Multi-stage build capability

2. **`docker-compose.yml`** - Local development environment
   - Main application service
   - Jupyter Lab service for notebooks
   - Volume mounting for data and models

### Testing Framework (`tests/`)
1. **`test_load_data.py`** - Data loading tests
2. **`test_preprocess.py`** - Data preprocessing tests
3. **`conftest.py`** - Pytest configuration
4. **`__init__.py`** - Package initialization

### Configuration Files
1. **`setup.cfg`** - Tool configurations (flake8, isort, pytest, coverage)
2. **`pyproject.toml`** - Modern Python project configuration
3. **`.gitignore`** - Updated with CI/CD artifacts and data files

### Documentation
1. **`.github/CICD.md`** - Comprehensive CI/CD documentation
2. **`.github/pull_request_template.md`** - PR template for consistency
3. **`setup.sh`** - Local development setup script

## 🚀 Quick Start

### 1. Local Development Setup
```bash
# Clone and navigate to repository
cd Grocry-Availability-Checker-And-Stock-Predictor

# Run setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8
```

### 2. Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### 3. Code Quality Checks
```bash
# Format code
black src/
isort src/

# Check linting
flake8 src/

# Run security scan
bandit -r src/
```

### 4. Docker Development
```bash
# Build and run
docker-compose up

# Access Jupyter Lab
open http://localhost:8888

# Run specific service
docker-compose up grocery-predictor
```

## 🔧 GitHub Setup

### Required Secrets
Add these in: Repository → Settings → Secrets and variables → Actions

1. **`DOCKER_USERNAME`** - Your Docker Hub username
2. **`DOCKER_PASSWORD`** - Your Docker Hub password/token

### Optional: Environment Protection
Configure in: Repository → Settings → Environments

1. Create `staging` environment
2. Create `production` environment
3. Add required reviewers for production
4. Set environment secrets if needed

## 📊 CI/CD Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     Developer Workflow                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│   Push to Branch (main, Aryan, aditya) / Create PR         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CI Pipeline (ci.yml)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │
│  │   Tests    │  │    Data    │  │   Code Quality     │   │
│  │ Multi-OS   │  │ Validation │  │  Black, Flake8,    │   │
│  │ Multi-Py   │  │            │  │  Pylint, Bandit    │   │
│  └────────────┘  └────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         Merge to Main → CD Pipeline (cd.yml)                │
│                                                              │
│  1. Build Docker Image → Push to Docker Hub                 │
│  2. Deploy to Staging                                        │
│  3. Deploy to Production (with approval)                     │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Testing Strategy

### Unit Tests
- Test data loading functionality
- Test preprocessing logic
- Test data transformations
- Mock large datasets for fast tests

### Integration Tests
- Test end-to-end data pipeline
- Test model training flow
- Test prediction functionality

### Coverage Goals
- Minimum 80% code coverage
- 100% coverage for critical paths
- Coverage reports in CI/CD

## 🛡️ Security Features

1. **Dependency Scanning** - Weekly updates and vulnerability checks
2. **Code Scanning** - Bandit for Python security issues
3. **Secret Management** - GitHub Secrets for sensitive data
4. **Container Scanning** - Docker image security (can be added)

## 📈 Monitoring and Alerts

### CI/CD Status
- Check GitHub Actions tab for build status
- Email notifications on failure
- Slack/Discord integration (can be configured)

### Code Coverage
- Codecov integration (configured in ci.yml)
- Coverage badges in README
- Historical coverage tracking

## 🔄 Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes and Test**
   ```bash
   # Write code
   # Run tests locally
   pytest tests/ -v
   # Format code
   black src/
   ```

3. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Use the PR template
   - Wait for CI checks to pass
   - Request review from team members

5. **Merge to Main**
   - Squash and merge
   - Automatic deployment triggered

## 📝 Best Practices

### Code Style
- Use Black for formatting (line length: 127)
- Use isort for import sorting
- Follow PEP 8 guidelines
- Add docstrings to functions

### Testing
- Write tests before pushing
- Aim for high coverage
- Use meaningful test names
- Mock external dependencies

### Git Workflow
- Use descriptive commit messages
- Keep commits atomic
- Rebase before merging
- Delete branches after merge

### Documentation
- Update README for new features
- Add inline comments for complex logic
- Keep CICD.md up to date
- Document API changes

## 🚨 Troubleshooting

### CI Failing?
1. Check Actions tab for error logs
2. Run tests locally to reproduce
3. Check Python version compatibility
4. Verify all dependencies are in requirements.txt

### Docker Build Issues?
1. Check Dockerfile syntax
2. Verify base image is available
3. Check for missing dependencies
4. Test build locally first

### Deployment Problems?
1. Verify secrets are configured
2. Check Docker Hub credentials
3. Review deployment logs
4. Verify environment settings

## 📞 Support

For issues or questions:
- Create an issue in GitHub
- Check `.github/CICD.md` for detailed docs
- Review workflow files in `.github/workflows/`

## 🎉 What's Next?

### Suggested Improvements
1. Add integration tests
2. Implement model versioning (MLflow/DVC)
3. Add API endpoints (FastAPI/Flask)
4. Set up monitoring (Prometheus/Grafana)
5. Add load testing
6. Implement A/B testing for models
7. Add automated rollback on failures

---

**Created**: November 2025  
**Team**: Aryan & Aditya  
**Project**: Grocery Availability Checker and Stock Predictor
