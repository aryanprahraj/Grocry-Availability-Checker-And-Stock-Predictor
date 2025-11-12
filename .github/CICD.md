# CI/CD Pipeline Documentation

## Overview
This repository uses GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD).

## Workflows

### 1. CI Pipeline (`ci.yml`)
Runs on every push and pull request to main, Aryan, and aditya branches.

**Jobs:**
- **Test**: Runs unit tests across multiple OS (Ubuntu, Windows, macOS) and Python versions (3.9, 3.10, 3.11)
- **Data Validation**: Validates data loading and preprocessing modules
- **Code Quality**: Checks code formatting, linting, security, and dependency vulnerabilities

**Key Features:**
- Multi-platform testing (Linux, Windows, macOS)
- Code coverage reporting with Codecov
- Linting with flake8
- Security scanning with Bandit
- Dependency vulnerability checking with Safety

### 2. CD Pipeline (`cd.yml`)
Runs on push to main branch or manual trigger.

**Jobs:**
- **Model Training**: Trains ML models (manual trigger only)
- **Docker Build & Push**: Builds and pushes Docker image to Docker Hub
- **Staging Deployment**: Deploys to staging environment
- **Production Deployment**: Deploys to production after staging success

**Required Secrets:**
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password or access token

### 3. Dependency Updates (`dependency-updates.yml`)
Runs weekly on Mondays at 9 AM UTC or manually.

**Features:**
- Automatically checks for dependency updates
- Creates a pull request with updated dependencies
- Helps keep dependencies secure and up-to-date

## Setup Instructions

### 1. Configure GitHub Secrets
Add the following secrets to your repository:
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password or access token

Go to: Repository → Settings → Secrets and variables → Actions → New repository secret

### 2. Enable GitHub Actions
- Go to Repository → Actions
- Enable workflows if not already enabled

### 3. Configure Environments (Optional)
For protected deployments:
- Go to Repository → Settings → Environments
- Create `staging` and `production` environments
- Add required reviewers for production

## Local Development

### Run Tests Locally
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Code Quality Checks
```bash
# Install quality tools
pip install black isort flake8 pylint bandit

# Format code
black src/
isort src/

# Check linting
flake8 src/

# Security scan
bandit -r src/
```

### Docker Development
```bash
# Build Docker image
docker build -t grocery-stock-predictor .

# Run with Docker Compose
docker-compose up

# Access Jupyter Lab
# Open browser: http://localhost:8888
```

## Deployment

### Manual Deployment Trigger
1. Go to Actions → CD Pipeline
2. Click "Run workflow"
3. Select branch and enable "Trigger model training" if needed
4. Click "Run workflow"

### Automatic Deployment
- Pushing to `main` branch triggers automatic deployment
- Staging deployment runs first
- Production deployment requires manual approval (if configured)

## Monitoring

### CI/CD Status
- Check the Actions tab for workflow status
- Green checkmark = All tests passed
- Red X = Tests failed, click for details

### Coverage Reports
- Coverage reports are uploaded to Codecov (if configured)
- HTML coverage reports available as artifacts

## Troubleshooting

### Tests Failing
1. Check the Actions tab for detailed error logs
2. Run tests locally to reproduce
3. Fix issues and push changes

### Deployment Issues
1. Verify all secrets are configured correctly
2. Check Docker Hub credentials
3. Review deployment logs in Actions tab

### Dependency Issues
1. Check for breaking changes in updated packages
2. Review dependency update PRs carefully
3. Run tests locally before merging

## Best Practices

1. **Always run tests locally** before pushing
2. **Keep dependencies updated** by reviewing weekly PRs
3. **Use feature branches** for development
4. **Merge via pull requests** with code review
5. **Monitor CI/CD pipelines** regularly
6. **Update documentation** when changing workflows
