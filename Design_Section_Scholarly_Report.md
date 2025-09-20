# 3. Design

## 3.1 Ideation and Design Planning

### Project Ideation

The Track My Impact system emerged from a critical analysis of waste management challenges in South Africa, where inadequate infrastructure and limited public awareness significantly impair recycling effectiveness (Godfrey et al. 2019). Early brainstorming sessions identified three primary design objectives: democratising waste classification through computer vision, providing immediate environmental feedback to users, and creating scalable infrastructure suitable for resource-constrained environments. The conceptual framework drew inspiration from behavioural change theory, particularly the Transtheoretical Model of behaviour change (Prochaska and DiClemente 1982), which emphasises the importance of immediate feedback and progressive engagement in sustaining environmental behaviours.

Initial concept refinement focused on addressing the fundamental disconnect between individual waste disposal actions and their environmental consequences. Research indicates that citizens often lack awareness of proper sorting techniques, with classification accuracy rates as low as 30-40% in domestic settings (Abos et al. 2024). The design team recognised that traditional educational approaches had proven insufficient, necessitating an interactive, technology-mediated solution that could provide real-time guidance and quantifiable impact metrics.

The ideation phase incorporated insights from environmental psychology literature, particularly the concept of "psychological distance" in environmental decision-making (Spence et al. 2012). By providing immediate, localised feedback on waste disposal choices, the system aims to reduce the temporal and spatial distance between individual actions and their environmental consequences, thereby enhancing user engagement and behaviour modification.

### Engagement with Stakeholders

Stakeholder engagement followed a structured approach involving three primary constituencies: potential end-users, domain experts, and technical specialists. Initial consultations with residents of Johannesburg townships revealed significant gaps in waste classification knowledge, particularly regarding mixed materials and electronic waste. Focus group discussions highlighted the necessity for multilingual support and low-data solutions, given the prevalence of prepaid mobile connections and limited internet infrastructure (Samson 2020).

Expert consultations with the National Cleaner Production Centre South Africa (NCPC-SA) provided crucial insights into local emission factors and waste processing capabilities. These discussions informed the decision to base environmental impact calculations on EPA WARM (Waste Reduction Model) version 15.2 data, adjusted for South African context using NCPC-SA emission factors (NCPC-SA 2021). The integration of authentic local data sources became a cornerstone of the system's credibility and relevance.

Technical stakeholder engagement included consultations with machine learning practitioners and mobile application developers familiar with resource-constrained environments. These discussions emphasised the importance of model optimisation and progressive web application (PWA) capabilities to ensure accessibility across diverse hardware configurations.

### Wireframes and Structural Planning

#### Frontend UI/UX Wireframes

The wireframing process employed a mobile-first design philosophy, recognising that smartphone penetration in South Africa exceeds desktop computer access (We Are Social 2023). Initial wireframes utilised a component-based approach, structuring the interface around discrete functional modules: waste capture, classification feedback, impact visualisation, and progress tracking.

```
Figure 3.1: Primary User Interface Wireframe Structure

┌─────────────────────────────────────┐
│ Header: Logo + User Status          │
├─────────────────────────────────────┤
│ Camera Capture Module               │
│ ┌─────────────┐ ┌─────────────────┐ │
│ │   Camera    │ │  Manual Entry   │ │
│ │   Button    │ │     Option      │ │
│ └─────────────┘ └─────────────────┘ │
├─────────────────────────────────────┤
│ Classification Results Display      │
│ ┌─────────────────────────────────┐ │
│ │ Material Type: Plastic Bottle   │ │
│ │ Confidence: 87.3%               │ │
│ │ Disposal Method: Recycling Bin  │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ Impact Metrics Visualisation        │
│ ┌─────────────────────────────────┐ │
│ │ CO₂ Saved: 147g                │ │
│ │ Water Saved: 2.53L             │ │
│ │ Energy Saved: 0.87 kWh         │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ Progress Tracking Dashboard         │
│ ┌─────────────────────────────────┐ │
│ │ Weekly Target: 12/15 items     │ │
│ │ Monthly Impact: 2.4 kg CO₂     │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

The wireframe structure prioritises immediate visual feedback whilst minimising cognitive load. Classification results are presented using a traffic-light colour system (green for high confidence, amber for medium, red for uncertain), drawing from established UX patterns in mobile health applications (Klasnja and Pratt 2012).

#### Information Architecture Diagrams

The information architecture follows a hierarchical structure designed to accommodate both novice and expert users. The primary navigation employs a bottom-tab pattern, providing access to four core modules: Classify, Impact, Progress, and Learn. This structure aligns with Nielsen's usability heuristics, particularly the principle of user control and freedom (Nielsen 1994).

```
Figure 3.2: Information Architecture Hierarchy

Track My Impact Application
├── Authentication Module
│   ├── Registration/Login
│   ├── User Profile Management
│   └── Privacy Settings
├── Classification Module
│   ├── Camera-based Input
│   ├── Manual Selection Interface
│   ├── Results Display
│   └── Alternative Suggestions
├── Impact Calculation Module
│   ├── Real-time Metrics Display
│   ├── Historical Tracking
│   └── Comparative Analytics
├── Progress Tracking Module
│   ├── Individual Dashboard
│   ├── Community Leaderboards
│   └── Achievement System
└── Educational Resources
    ├── Waste Classification Guide
    ├── Local Recycling Information
    └── Environmental Impact Education
```

The modular architecture enables progressive disclosure of information, allowing users to engage with the system at their preferred level of complexity whilst maintaining pathways to deeper engagement (Lidwell et al. 2010).

#### High-level Flowcharts for User Journey and Data Flow

The user journey flowchart maps the complete interaction sequence from initial system access through waste classification to impact feedback. The design incorporates decision points that accommodate varying levels of user confidence and technical proficiency.

```
Figure 3.3: Primary User Journey Flowchart

START
  ↓
[Authentication Check]
  ↓
[New User?] ──Yes──→ [Onboarding Tutorial]
  ↓ No                     ↓
[Main Dashboard] ←─────────┘
  ↓
[Select Classification Method]
  ├─ Camera Input ──→ [Image Capture] ──→ [CNN Processing]
  └─ Manual Entry ──→ [Material Selection] ──→ [Direct Classification]
  ↓
[Classification Results]
  ↓
[Confidence > 80%?] ──No──→ [Show Alternatives] ──→ [User Confirmation]
  ↓ Yes                                              ↓
[Weight Estimation] ←───────────────────────────────┘
  ↓
[Impact Calculation] ──→ [WARM Factor Application]
  ↓
[Results Display] ──→ [Update User Statistics]
  ↓
[Gamification Feedback] ──→ [Achievement Check]
  ↓
[Return to Dashboard]
```

The data flow architecture implements a separation of concerns principle, isolating the machine learning pipeline from user interface logic to ensure system maintainability and scalability (Shklar and Rosen 2009).

### Project Management

#### Task Breakdown and Sprint Planning

The development process employed Agile methodology with two-week sprint cycles, structured around technical milestones and user validation checkpoints. The initial phase focused on establishing the core classification pipeline, whilst subsequent sprints incrementally added user interface components and advanced features.

**Table 3.1: Sprint Structure and Key Deliverables**

| Sprint | Duration | Primary Objectives | Key Deliverables | Dependencies |
|--------|----------|-------------------|------------------|--------------|
| S0 | Week 0 | Environment Setup | Python/Node.js scaffolding, CI/CD pipeline | None |
| S1 | Week 1 | Data Preparation | Kaggle dataset split, initial EDA | Kaggle API access |
| S2-3 | Weeks 2-3 | CNN Development | Trained classification model, validation metrics | S1 completion |
| S4 | Week 4 | Prototype Impact System | Emission factors integration, basic calculations | CNN baseline |
| S5 | Week 5 | Domestic Material Mapping | SA-specific waste categories, WARM integration | Domain expert consultation |
| S6-7 | Weeks 6-7 | WARM v15.2 Integration | Processed impact factors, validation framework | EPA data access |
| S8 | Week 8 | Analytics Dashboard | Streamlit application, visualisation components | Processed data pipeline |
| S9-10 | Weeks 9-10 | API Development | FastAPI backend, model conversion to TFLite | Model training completion |
| S11 | Week 11 | Database Design | SQLAlchemy models, data consistency protocols | API foundation |
| S12 | Week 12 | Frontend Integration | Next.js application, component development | Backend API availability |
| S13 | Week 13 | Deployment and Testing | Production deployment, user acceptance testing | Full system integration |

#### Milestones and Deliverable Scheduling

Key project milestones were established to ensure alignment between technical development and user requirements validation. Each milestone included specific success criteria and rollback procedures to manage development risk.

**Milestone 1: Functional CNN Prototype (Week 3)**
- Success Criteria: Classification accuracy ≥ 75% on validation set, inference time < 500ms
- Deliverables: Trained model artifacts, performance benchmarks, preliminary confusion matrix analysis
- Risk Mitigation: Alternative model architectures pre-evaluated for fallback implementation

**Milestone 2: Impact Calculation Framework (Week 7)**
- Success Criteria: WARM factor integration complete, unit test coverage ≥ 90%
- Deliverables: Impact calculation API, domestic material mapping, validation test suite
- Risk Mitigation: Simplified impact model prepared as reduced-scope alternative

**Milestone 3: Full-Stack Integration (Week 12)**
- Success Criteria: End-to-end user flow functional, API response times < 200ms
- Deliverables: Deployed application, user interface components, authentication system
- Risk Mitigation: Progressive web app (PWA) implementation to reduce deployment complexity

#### Risk and Contingency Planning

Risk assessment identified three primary threat categories: technical implementation challenges, data quality issues, and user adoption barriers. Each category received dedicated mitigation strategies and contingency protocols.

**Technical Risks:**
- Model Performance Degradation: Contingency planning included ensemble methods and human-in-the-loop validation for uncertain classifications
- Infrastructure Scalability: Cloud-native deployment strategy with container orchestration to handle variable user loads
- Device Compatibility: Progressive enhancement approach ensuring core functionality across diverse hardware configurations

**Data Quality Risks:**
- Training Data Bias: Systematic validation against South African waste stream composition data (DEA 2018)
- Impact Factor Accuracy: Cross-validation with multiple environmental databases including Our World in Data and local NCPC-SA publications
- User Input Variability: Robust preprocessing pipeline with data augmentation to handle diverse image quality and lighting conditions

**User Adoption Risks:**
- Digital Literacy Barriers: Simplified interface design with visual cues and minimal text requirements
- Connectivity Constraints: Offline-first architecture with local storage and synchronisation capabilities
- Cultural Relevance: Stakeholder feedback integration and iterative design refinement based on user testing

## 3.2 System Architecture and Data Model

### Monorepo Structure

The system architecture implements a distributed monorepo pattern, facilitating coordinated development across multiple technology stacks whilst maintaining clear separation of concerns (Kong 2005). The architectural decision reflects best practices in modern software engineering, where microservice principles are applied within a unified codebase to maximise development velocity whilst ensuring system maintainability.

```
Figure 3.4: Monorepo Directory Structure

track-my-impact/
├── backend/                          # FastAPI microservice
│   ├── app/
│   │   ├── models/
│   │   │   ├── database_models.py    # SQLAlchemy entities
│   │   │   └── schemas.py            # Pydantic DTOs
│   │   ├── routers/
│   │   │   ├── auth.py              # Authentication endpoints
│   │   │   ├── waste.py             # Classification API
│   │   │   └── impact.py            # Impact calculation API
│   │   ├── core/
│   │   │   ├── config.py            # Environment configuration
│   │   │   ├── database.py          # Database connection management
│   │   │   └── security.py          # Authentication middleware
│   │   └── utils/
│   │       ├── impact_calculator.py  # WARM factor application
│   │       └── model_loader.py       # TFLite model management
│   ├── scripts/
│   │   ├── deploy.sh                # Deployment automation
│   │   └── seed_data.py             # Database initialisation
│   └── tests/
│       ├── test_classification.py   # API endpoint testing
│       └── test_impact_calculation.py # Impact calculation validation
├── frontend/                         # Next.js application
│   ├── app/
│   │   ├── (auth)/                  # Authentication routes
│   │   ├── classify/                # Classification interface
│   │   ├── dashboard/               # User progress tracking
│   │   └── impact/                  # Impact visualisation
│   ├── components/
│   │   ├── ui/                      # shadcn/ui components
│   │   ├── waste-classifier/        # Classification UI components
│   │   └── impact-display/          # Metrics visualisation components
│   ├── lib/
│   │   ├── api.ts                   # API client configuration
│   │   ├── auth.ts                  # Authentication utilities
│   │   └── types.ts                 # TypeScript type definitions
│   └── public/
│       ├── models/                  # TFLite model artifacts
│       ├── images/                  # Static assets
│       └── data/                    # Reference datasets
└── shared/
    ├── data/
    │   ├── domestic_materials_processed.csv  # Material definitions
    │   ├── warm_factors_processed.csv        # Impact calculation factors
    │   └── petco_locations.csv               # Recycling facility data
    └── docs/
        ├── API.md                   # API documentation
        ├── DEPLOYMENT.md            # Infrastructure guide
        └── MODEL_CONVERSION.md      # Model optimisation procedures
```

This structure adheres to Clean Architecture principles, isolating business logic from infrastructure concerns and enabling independent testing and deployment of system components (Martin 2017). The separation facilitates parallel development streams whilst ensuring consistent data contracts through shared schema definitions.

### Components: Next.js/React Frontend, FastAPI Backend, Dockerised Orchestration

#### Frontend Architecture (Next.js/React)

The frontend implementation utilises Next.js 13+ with the App Router architecture, providing server-side rendering capabilities essential for performance in bandwidth-constrained environments. The component hierarchy follows atomic design principles, with reusable UI elements constructed using shadcn/ui components for consistent styling and accessibility compliance.

```typescript
// Example: Core Classification Component Architecture
interface ClassificationComponentProps {
  onClassificationComplete: (result: ClassificationResult) => void;
  allowManualEntry?: boolean;
  maxImageSize?: number;
}

const WasteClassifier: React.FC<ClassificationComponentProps> = ({
  onClassificationComplete,
  allowManualEntry = true,
  maxImageSize = 2048
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  const handleImageCapture = useCallback(async (imageFile: File) => {
    setIsProcessing(true);
    try {
      const result = await classifyWaste(imageFile);
      onClassificationComplete(result);
    } catch (error) {
      // Error handling with user-friendly messaging
      showToast('Classification failed. Please try again.', 'error');
    } finally {
      setIsProcessing(false);
    }
  }, [onClassificationComplete]);

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle>Classify Your Waste</CardTitle>
      </CardHeader>
      <CardContent>
        <ImageCapture
          onImageSelected={handleImageCapture}
          disabled={isProcessing}
          maxSize={maxImageSize}
        />
        {allowManualEntry && (
          <ManualEntryFallback
            onMaterialSelected={onClassificationComplete}
          />
        )}
      </CardContent>
    </Card>
  );
};
```

The frontend architecture emphasises progressive enhancement, ensuring core functionality remains accessible even when advanced features (such as camera access) are unavailable. This approach aligns with inclusive design principles and accommodates the diverse hardware landscape in South African markets.

#### Backend Architecture (FastAPI)

The backend service implements a RESTful API using FastAPI, chosen for its automatic OpenAPI documentation generation and high-performance characteristics suitable for machine learning inference workloads. The architecture follows Domain-Driven Design principles, with clear boundaries between authentication, classification, and impact calculation domains.

```python
# Example: Classification Router Implementation
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.schemas import ClassificationRequest, ClassificationResult
from app.utils.model_loader import get_classification_model
from app.utils.impact_calculator import EnvironmentalImpactCalculator

router = APIRouter(prefix="/api/waste", tags=["waste"])

@router.post("/classify", response_model=ClassificationResult)
async def classify_waste_item(
    request: ClassificationRequest,
    image: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Classify waste item via image upload or manual selection.

    Returns classification result with confidence scores,
    material properties, and disposal recommendations.
    """
    try:
        if image:
            # Image-based classification
            model = get_classification_model()
            prediction = await model.predict(image)

            # Map CNN output to domestic material
            material = db.query(DomesticMaterial).filter(
                DomesticMaterial.cnn_class_name == prediction.class_name
            ).first()

            if not material:
                raise HTTPException(
                    status_code=404,
                    detail=f"Material mapping not found for {prediction.class_name}"
                )

            return ClassificationResult(
                material_id=material.material_id,
                cnn_class_name=prediction.class_name,
                display_name=material.domestic_definition,
                confidence=prediction.confidence,
                typical_weight_grams=material.typical_weight_grams,
                warm_category=material.warm_category,
                alternative_predictions=prediction.alternatives
            )

        else:
            # Manual selection pathway
            material = db.query(DomesticMaterial).filter(
                DomesticMaterial.material_id == request.material_id
            ).first()

            if not material:
                raise HTTPException(status_code=404, detail="Material not found")

            return ClassificationResult(
                material_id=material.material_id,
                display_name=material.domestic_definition,
                confidence=1.0,  # User-confirmed selection
                typical_weight_grams=material.typical_weight_grams,
                warm_category=material.warm_category
            )

    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Classification service temporarily unavailable"
        )

@router.post("/calculate-impact")
async def calculate_environmental_impact(
    material_id: str,
    weight_grams: float,
    disposal_method: DisposalMethod,
    db: Session = Depends(get_db)
):
    """Calculate environmental impact using WARM factors."""
    calculator = EnvironmentalImpactCalculator(db)
    impact = calculator.calculate_impact(
        material_id=material_id,
        weight_grams=weight_grams,
        disposal_method=disposal_method
    )
    return impact
```

The API design prioritises defensive programming practices, with comprehensive error handling and input validation to ensure system stability under diverse usage patterns.

#### Dockerised Orchestration

The deployment architecture utilises containerisation to ensure consistency across development, staging, and production environments. The Docker Compose configuration orchestrates multiple services whilst maintaining development workflow efficiency.

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/trackimpact
      - CORS_ORIGINS=http://localhost:3000
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - db
    volumes:
      - ./shared/data:/app/data
      - ./backend/models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_APP_ENV=development
    depends_on:
      - backend
    volumes:
      - ./frontend:/app
      - /app/node_modules

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=trackimpact
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backend/scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

This orchestration strategy enables rapid development iteration whilst providing production-ready deployment capabilities through environment-specific configuration management.

### Database Design

#### Entity-Relationship Diagram and Schema Rationale

The database schema implements a normalised structure designed to support both operational requirements and analytical workloads. The design separates user data from reference data to enable independent scaling and maintenance cycles.

```
Figure 3.5: Entity-Relationship Diagram

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      User       │    │   UserSession   │    │   UserImpact    │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ user_id (PK)    │──┐ │ session_id (PK) │    │ impact_id (PK)  │
│ email           │  │ │ user_id (FK)    │──┐ │ user_id (FK)    │
│ username        │  │ │ created_at      │  │ │ material_id (FK)│
│ created_at      │  │ │ expires_at      │  │ │ weight_grams    │
│ last_login      │  │ │ is_active       │  │ │ disposal_method │
│ preferences     │  │ └─────────────────┘  │ │ co2_saved_kg    │
└─────────────────┘  │                     │ │ created_at      │
         │           │                     │ └─────────────────┘
         └───────────┘                     │          │
                                          │          │
┌─────────────────┐    ┌─────────────────┐ │          │
│ DomesticMaterial│    │   WarmFactor    │ │          │
├─────────────────┤    ├─────────────────┤ │          │
│ material_id (PK)│──┐ │ factor_id (PK)  │ │          │
│ cnn_class_name  │  │ │ warm_category   │ │          │
│ domestic_defn   │  │ │ disposal_method │ │          │
│ material_type   │  │ │ co2e_kg_per_ton │ │          │
│ warm_category   │──┼─│ energy_kwh_per_ton│         │
│ density_kg_m3   │  │ │ water_l_per_ton │ │          │
│ typical_weight  │  │ │ created_at      │ │          │
│ created_at      │  │ └─────────────────┘ │          │
└─────────────────┘  │                     │          │
         │           │                     │          │
         └───────────┴─────────────────────┘          │
                                                     │
┌─────────────────┐    ┌─────────────────┐           │
│ Classification  │    │   Achievement   │           │
├─────────────────┤    ├─────────────────┤           │
│ classification_id│    │ achievement_id  │           │
│ user_id (FK)    │──┐ │ user_id (FK)    │           │
│ material_id (FK)│  │ │ type            │           │
│ confidence      │  │ │ criteria_met    │           │
│ image_url       │  │ │ awarded_at      │           │
│ created_at      │  │ └─────────────────┘           │
└─────────────────┘  │          │                   │
         │           │          └───────────────────┘
         └───────────┘
```

#### Storage Rationale and Data Consistency Protocols

The schema design prioritises data integrity through foreign key constraints whilst enabling performant queries for both real-time classification and analytical reporting. Key design decisions include:

**Material Reference Data Separation:** The `DomesticMaterial` and `WarmFactor` tables contain relatively static reference data, enabling independent versioning and updates without affecting user transaction data.

**Denormalised Impact Storage:** Pre-calculated impact metrics are stored in `UserImpact` to optimise dashboard query performance, following the principle that read-heavy workloads benefit from strategic denormalisation (Kleppmann 2017).

**Temporal Data Tracking:** All entities include timestamp fields to support auditing and temporal analysis requirements.

```sql
-- Example: Database Schema Creation with Constraints
CREATE TABLE domestic_materials (
    material_id VARCHAR(32) PRIMARY KEY,
    cnn_class_name VARCHAR(100) NOT NULL UNIQUE,
    domestic_definition TEXT NOT NULL,
    material_type VARCHAR(50) NOT NULL,
    warm_category VARCHAR(100) NOT NULL,
    density_kg_per_m3 DECIMAL(8,2),
    typical_weight_grams INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_cnn_class (cnn_class_name),
    INDEX idx_warm_category (warm_category),
    INDEX idx_material_type (material_type)
);

CREATE TABLE warm_factors (
    factor_id SERIAL PRIMARY KEY,
    warm_category VARCHAR(100) NOT NULL,
    disposal_method ENUM('recycle', 'landfill', 'incineration', 'compost') NOT NULL,
    co2e_kg_per_ton DECIMAL(12,6) NOT NULL,
    energy_kwh_per_ton DECIMAL(12,6),
    water_litres_per_ton DECIMAL(12,6),
    methane_emissions_kg_per_ton DECIMAL(12,6),
    carbon_storage_kg_per_ton DECIMAL(12,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_category_method (warm_category, disposal_method),
    INDEX idx_warm_category (warm_category)
);

CREATE TABLE user_impacts (
    impact_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    material_id VARCHAR(32) NOT NULL,
    weight_grams DECIMAL(8,2) NOT NULL,
    disposal_method ENUM('recycle', 'landfill', 'incineration', 'compost') NOT NULL,
    co2_saved_kg DECIMAL(12,6) NOT NULL,
    water_saved_litres DECIMAL(12,6),
    energy_saved_kwh DECIMAL(12,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (material_id) REFERENCES domestic_materials(material_id),
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_material (material_id)
);
```

#### Data Consistency and Integrity Protocols

Data consistency is maintained through multiple complementary mechanisms:

**Transactional Integrity:** All multi-table operations utilise database transactions to ensure atomicity, particularly for user impact logging that updates both classification and impact tables simultaneously.

**Referential Integrity:** Foreign key constraints prevent orphaned records and maintain relational consistency, with appropriate cascade rules for user data deletion whilst preserving reference data integrity.

**Data Validation:** Application-level validation using Pydantic schemas ensures type safety and business rule compliance before database persistence.

```python
# Example: Transactional Impact Logging
@router.post("/log-impact")
async def log_waste_impact(
    classification_result: ClassificationResult,
    weight_grams: float,
    disposal_method: DisposalMethod,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Log environmental impact with transactional consistency."""

    async with db.begin():
        try:
            # Calculate impact using WARM factors
            impact_calc = EnvironmentalImpactCalculator(db)
            impact_metrics = impact_calc.calculate_impact(
                material_id=classification_result.material_id,
                weight_grams=weight_grams,
                disposal_method=disposal_method
            )

            # Create classification record
            classification = Classification(
                user_id=current_user.user_id,
                material_id=classification_result.material_id,
                confidence=classification_result.confidence,
                created_at=datetime.utcnow()
            )
            db.add(classification)

            # Create impact record
            user_impact = UserImpact(
                user_id=current_user.user_id,
                material_id=classification_result.material_id,
                weight_grams=weight_grams,
                disposal_method=disposal_method,
                co2_saved_kg=impact_metrics.co2_saved_kg,
                water_saved_litres=impact_metrics.water_saved_litres,
                energy_saved_kwh=impact_metrics.energy_saved_kwh,
                created_at=datetime.utcnow()
            )
            db.add(user_impact)

            # Update user statistics (denormalised for performance)
            await update_user_statistics(current_user.user_id, impact_metrics)

            await db.commit()
            return {"status": "success", "impact": impact_metrics}

        except Exception as e:
            await db.rollback()
            logger.error(f"Impact logging failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Impact logging failed")
```

## 3.3 User Experience and Education

### User Journey Mapping

#### Typical User Paths

The user journey design accommodates multiple entry points and engagement levels, recognising that waste management behaviour change occurs through progressive commitment rather than immediate transformation (Prochaska and DiClemente 1982). Five primary user paths have been identified through stakeholder research and usability testing:

**Path 1: First-Time User (Registration and Onboarding)**
```
Entry Point → App Download/Web Access
     ↓
Registration (email/social auth)
     ↓
Privacy Consent and Preferences
     ↓
Interactive Tutorial (waste classification basics)
     ↓
First Classification Attempt (guided)
     ↓
Impact Feedback and Goal Setting
     ↓
Dashboard Familiarisation
```

**Path 2: Regular User (Daily Waste Logging)**
```
App Launch → Quick Authentication
     ↓
Dashboard Overview (weekly progress)
     ↓
Camera Activation or Manual Entry
     ↓
Waste Classification (2-3 items average)
     ↓
Immediate Impact Feedback
     ↓
Progress Update and Streak Maintenance
     ↓
Optional: Community Comparison
```

**Path 3: Uncertain User (Low-Confidence Classifications)**
```
Image Capture → CNN Processing
     ↓
Low Confidence Result (< 70%)
     ↓
Alternative Suggestions Display
     ↓
Educational Context and Tips
     ↓
Manual Confirmation or Correction
     ↓
Learning Algorithm Update
     ↓
Improved Guidance for Similar Items
```

**Path 4: Educational User (Resource Access and Learning)**
```
Dashboard → Educational Resources Tab
     ↓
Local Recycling Information
     ↓
Material-Specific Guidelines
     ↓
Environmental Impact Education
     ↓
Quiz and Knowledge Validation
     ↓
Achievement Badge Award
     ↓
Community Sharing Options
```

**Path 5: Community User (Social Engagement and Competition)**
```
Dashboard → Community Features
     ↓
Leaderboard View (neighbourhood/city)
     ↓
Challenge Participation
     ↓
Impact Comparison and Discussion
     ↓
Achievement Sharing
     ↓
Peer Learning and Motivation
```

Each pathway includes carefully designed exit points and re-engagement triggers, ensuring users can temporarily disengage without losing progress or motivation to return.

#### Journey Optimisation and Friction Reduction

User journey optimisation focuses on minimising cognitive load whilst maximising educational value. Key design decisions include:

**Progressive Disclosure:** Advanced features are introduced gradually, with core functionality accessible within three taps from any entry point.

**Context-Aware Assistance:** Help content and suggestions adapt based on user confidence levels and historical classification patterns.

**Graceful Degradation:** All core features remain functional even when advanced capabilities (camera, GPS, internet connectivity) are unavailable.

```typescript
// Example: Adaptive User Interface Logic
interface UserJourneyState {
  isFirstTimeUser: boolean;
  classificationCount: number;
  averageConfidence: number;
  preferredInputMethod: 'camera' | 'manual';
  lastActiveDate: Date;
}

const getAdaptiveInterface = (journeyState: UserJourneyState): InterfaceConfig => {
  if (journeyState.isFirstTimeUser) {
    return {
      showTutorialPrompts: true,
      enableGuidedMode: true,
      simplifyNavigation: true,
      highlightPrimaryActions: true
    };
  }

  if (journeyState.averageConfidence < 0.7) {
    return {
      showAlternativeSuggestions: true,
      enableEducationalTooltips: true,
      prioritiseManualEntry: true,
      expandExplanations: true
    };
  }

  if (journeyState.classificationCount > 50) {
    return {
      enableAdvancedFeatures: true,
      showCommunityFeatures: true,
      enableBatchMode: true,
      displayDetailedAnalytics: true
    };
  }

  return defaultInterfaceConfig;
};
```

### Behavioural UX Motivators

#### Visual and Auditory Feedback Systems

The feedback system employs principles from operant conditioning and positive reinforcement to encourage sustained engagement (Skinner 1953). Visual feedback utilises colour psychology and progressive visual enhancement to communicate success and progress.

**Immediate Classification Feedback:**
- High confidence (≥ 80%): Green checkmark with positive sound cue
- Medium confidence (60-79%): Amber warning with neutral notification
- Low confidence (< 60%): Red caution with educational prompt

**Progressive Impact Visualisation:**
The system employs incremental impact display, showing immediate item-level feedback followed by contextualised cumulative effects:

```javascript
// Example: Progressive Impact Feedback Implementation
const ImpactFeedback = ({ impactData, animationDelay = 0 }) => {
  const [displayStage, setDisplayStage] = useState(0);

  useEffect(() => {
    const stages = [
      { delay: 0, content: 'immediate' },      // Item impact
      { delay: 1000, content: 'daily' },       // Daily total
      { delay: 2000, content: 'weekly' },      // Weekly progress
      { delay: 3000, content: 'contextual' }   // Environmental context
    ];

    stages.forEach((stage, index) => {
      setTimeout(() => setDisplayStage(index), stage.delay + animationDelay);
    });
  }, [impactData, animationDelay]);

  return (
    <div className="impact-feedback-container">
      <AnimatePresence mode="wait">
        {displayStage >= 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
          >
            <ImpactCard
              title="This Item"
              metrics={{
                co2: `${impactData.co2_saved_kg}kg CO₂ saved`,
                water: `${impactData.water_saved_litres}L water conserved`,
                energy: `${impactData.energy_saved_kwh}kWh energy recovered`
              }}
              icon="🌱"
            />
          </motion.div>
        )}

        {displayStage >= 1 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.4 }}
          >
            <ProgressRing
              current={impactData.dailyProgress}
              target={impactData.dailyTarget}
              label="Today's Impact"
            />
          </motion.div>
        )}

        {displayStage >= 2 && (
          <ContextualMessage
            message={generateContextualMessage(impactData)}
            type="environmental"
          />
        )}
      </AnimatePresence>
    </div>
  );
};
```

#### Milestone "Nudges" and Achievement Systems

The achievement system implements Variable Ratio Reinforcement Schedule principles, providing unpredictable positive reinforcement to maintain user engagement over extended periods (Ferster and Skinner 1957).

**Achievement Categories:**
1. **Classification Mastery:** Accuracy-based rewards encouraging learning
2. **Consistency Streaks:** Daily/weekly engagement maintenance
3. **Environmental Impact:** Cumulative CO₂ savings milestones
4. **Community Contribution:** Peer education and assistance
5. **Local Engagement:** Recycling facility visits and local action

```typescript
// Example: Achievement System Implementation
interface Achievement {
  id: string;
  title: string;
  description: string;
  criteria: AchievementCriteria;
  reward: AchievementReward;
  isVisible: boolean;
  progressTracking: boolean;
}

const achievementDefinitions: Achievement[] = [
  {
    id: 'first_classification',
    title: 'Waste Detective',
    description: 'Complete your first waste classification',
    criteria: { classificationsCount: 1 },
    reward: { badgeIcon: '🔍', experiencePoints: 10 },
    isVisible: true,
    progressTracking: false
  },
  {
    id: 'accuracy_master',
    title: 'Sorting Expert',
    description: 'Achieve 90% accuracy over 20 classifications',
    criteria: {
      minClassifications: 20,
      averageAccuracy: 0.9
    },
    reward: { badgeIcon: '🎯', experiencePoints: 100 },
    isVisible: false, // Hidden until prerequisites met
    progressTracking: true
  },
  {
    id: 'co2_saver_1kg',
    title: 'Carbon Warrior',
    description: 'Save 1kg of CO₂ through proper recycling',
    criteria: { cumulativeCO2Saved: 1.0 },
    reward: {
      badgeIcon: '🌍',
      experiencePoints: 50,
      unlocks: ['advanced_analytics']
    },
    isVisible: true,
    progressTracking: true
  }
];

const checkAchievements = async (userId: string, userStats: UserStatistics) => {
  const unlockedAchievements = [];

  for (const achievement of achievementDefinitions) {
    if (await isAchievementUnlocked(userId, achievement.id)) continue;

    const isEligible = evaluateAchievementCriteria(achievement.criteria, userStats);

    if (isEligible) {
      await unlockAchievement(userId, achievement);
      unlockedAchievements.push(achievement);

      // Trigger celebration animation and notification
      await triggerAchievementCelebration(achievement);
    }
  }

  return unlockedAchievements;
};
```

#### Educational Pop-ups and Contextual Learning

Educational interventions are delivered through contextually appropriate micro-learning moments, designed to enhance user knowledge without disrupting primary task completion (Clark and Mayer 2016).

**Adaptive Educational Content:**
The system personalises educational content based on user classification patterns, confidence levels, and regional recycling infrastructure availability.

```typescript
// Example: Contextual Education System
interface EducationalContext {
  userClassificationHistory: Classification[];
  currentMaterial: DomesticMaterial;
  localRecyclingOptions: RecyclingFacility[];
  confidenceLevel: number;
}

const generateEducationalContent = (context: EducationalContext): EducationalPopup => {
  const { currentMaterial, confidenceLevel, localRecyclingOptions } = context;

  // Low confidence - provide material identification tips
  if (confidenceLevel < 0.6) {
    return {
      type: 'identification_tips',
      title: `Identifying ${currentMaterial.material_type}`,
      content: generateIdentificationTips(currentMaterial),
      actionButton: {
        text: 'Learn More',
        action: () => navigateToMaterialGuide(currentMaterial.material_id)
      },
      dismissable: true,
      showFrequency: 'occasionally' // Show 30% of the time for this confidence level
    };
  }

  // High confidence - provide impact context
  if (confidenceLevel > 0.8 && hasNearbyRecyclingOptions(localRecyclingOptions)) {
    return {
      type: 'impact_context',
      title: 'Your Impact Matters',
      content: generateImpactContext(currentMaterial, localRecyclingOptions),
      actionButton: {
        text: 'Find Recycling Centres',
        action: () => showNearbyRecyclingMap(localRecyclingOptions)
      },
      dismissable: true,
      showFrequency: 'rarely' // Show 10% of the time for engagement
    };
  }

  // Default case - material-specific tips
  return generateMaterialSpecificTips(currentMaterial);
};

const generateIdentificationTips = (material: DomesticMaterial): string => {
  const tips = {
    plastic: [
      "Look for recycling codes (numbers 1-7) inside the triangle symbol",
      "PET bottles (code 1) are clear and lightweight",
      "HDPE containers (code 2) are often opaque and include milk jugs"
    ],
    glass: [
      "Clear glass reflects light and feels heavier than plastic",
      "Coloured glass (brown, green) often contains beverages or medicines",
      "Remove caps and lids before recycling"
    ],
    paper: [
      "Clean, dry paper recycles best",
      "Remove plastic windows from envelopes",
      "Cardboard should be flattened to save space"
    ]
  };

  return tips[material.material_type]?.[Math.floor(Math.random() * tips[material.material_type].length)]
    || "Check local recycling guidelines for specific requirements";
};
```

The educational system emphasises micro-learning principles, delivering knowledge in digestible segments that respect user cognitive load whilst building comprehensive understanding over time.

---

## References

Abos, Francesca, et al. "Waste Management Behaviour Change Interventions: A Systematic Review." *Environmental Science & Policy*, vol. 145, 2024, pp. 67-81.

Clark, Richard C., and Richard E. Mayer. *E-Learning and the Science of Instruction: Proven Guidelines for Consumers and Designers of Multimedia Learning*. 4th ed., Wiley, 2016.

Department of Environmental Affairs (DEA). *South Africa's Waste Information System*. DEA, 2018.

Ferster, Charles B., and B.F. Skinner. *Schedules of Reinforcement*. Appleton-Century-Crofts, 1957.

Fielding, Roy Thomas. *Architectural Styles and the Design of Network-based Software Architectures*. Dissertation, University of California, Irvine, 2000.

Godfrey, Linda, et al. "The Current Status of Waste Management in South Africa." *Waste Management*, vol. 84, 2019, pp. 120-135.

Klasnja, Predrag, and Wanda Pratt. "Healthcare in the Pocket: Mapping the Space of Mobile-Phone Health Interventions." *Journal of Biomedical Informatics*, vol. 45, no. 1, 2012, pp. 184-198.

Kleppmann, Martin. *Designing Data-Intensive Applications*. O'Reilly Media, 2017.

Kong, Xiangzhen. "Separation of Concerns: A Web Application Architecture Framework." *IEEE International Conference on Services Computing*, 2005, pp. 443-450.

Lidwell, William, et al. *Universal Principles of Design*. Rockport Publishers, 2010.

Martin, Robert C. *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall, 2017.

National Cleaner Production Centre South Africa (NCPC-SA). *Green Economy Transition and Carbon Footprint Assessment*. NCPC-SA, 2021.

Nielsen, Jakob. "Heuristic Evaluation of User Interfaces." *CHI '90: Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*, 1994, pp. 249-256.

Our World in Data (OWID). *Environmental Impact of Food Production*. OWID, 2023, www.ourworldindata.org/environmental-impacts-food.

Prochaska, James O., and Carlo C. DiClemente. "Transtheoretical Therapy: Toward a More Integrative Model of Change." *Psychotherapy: Theory, Research & Practice*, vol. 19, no. 3, 1982, pp. 276-288.

Samson, Melanie. "Accumulation by Dispossession and the Informal Economy – Struggles over Knowledge, Being and Waste at a Johannesburg Dump." *Environment and Planning A: Economy and Space*, vol. 52, no. 8, 2020, pp. 1527-1544.

Shklar, Leon, and Rich Rosen. *Web Application Architecture: Principles, Protocols and Practices*. 2nd ed., Wiley, 2009.

Skinner, B.F. *Science and Human Behavior*. Macmillan, 1953.

Spence, Alexa, et al. "The Psychological Distance of Climate Change." *Risk Analysis*, vol. 32, no. 6, 2012, pp. 957-972.

We Are Social. *Digital 2023: South Africa*. DataReportal, 2023.

Yardley, Lucy, et al. "The Person-Based Approach to Intervention Development: Application to Digital Health-Related Behavior Change Interventions." *Journal of Medical Internet Research*, vol. 18, no. 1, 2016, e30.
