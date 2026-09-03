from pathlib import Path
import re

from backend.app.core.config import Settings, get_settings
from backend.app.core.paths import dataset_paths, resolve_project_path
from backend.app.db.schemas import (
    DatasetClassSummary,
    DatasetDiscoveryResponse,
    DatasetPathItem,
    DatasetPathsResponse,
    DatasetSummary,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = IMAGE_EXTENSIONS
COMPANY_NORMAL_FOLDER_NAMES = {"good", "normal", "non defective", "non_defective", "nondefective", "nondefects", "ok"}


def _count_files(path: Path, extensions: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(1 for file_path in path.rglob("*") if file_path.is_file() and file_path.suffix.lower() in extensions)


def _path_string(path: Path) -> str:
    return str(path)


class DatasetService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def list_configured_paths(self) -> dict[str, Path]:
        return dataset_paths(self.settings)

    def get_configured_paths(self) -> DatasetPathsResponse:
        return DatasetPathsResponse(
            items=[
                DatasetPathItem(name=name, path=_path_string(path), exists=path.exists())
                for name, path in self.list_configured_paths().items()
            ]
        )

    def discover(self) -> DatasetDiscoveryResponse:
        paths = self.list_configured_paths()
        datasets = [
            self._discover_company(paths["raw_company"]),
            self._discover_dme(paths["dme_20250129"]),
            self._discover_wfdd(paths["wfdd"]),
            self._discover_lusitano(paths["lusitano"]),
            self._discover_mvtec(paths["mvtec_ad"]),
            self._discover_reference_ready(paths["reference_ready"]),
        ]
        return DatasetDiscoveryResponse(
            datasets=datasets,
            total_images=sum(item.image_count for dataset in datasets for item in dataset.classes),
        )

    def resolve_reference_normal_path(self, dataset_name: str, fabric_type: str | None, normal_path: str | None) -> Path | None:
        if normal_path:
            candidate = Path(normal_path)
            return candidate if candidate.is_absolute() else resolve_project_path(normal_path)

        paths = self.list_configured_paths()
        normalized_name = dataset_name.lower()
        if normalized_name in {"dme", "dme_20250129", "20250129_fabric_defects"}:
            return paths["dme_20250129"] / "non defective"
        if normalized_name == "wfdd" and fabric_type:
            return paths["wfdd"] / fabric_type / "train" / "good"
        if normalized_name == "lusitano":
            return paths["lusitano"] / "nondefects"
        if normalized_name in {"company", "company_iqc"}:
            normal_dirs = self._company_normal_dirs(paths["raw_company"])
            return normal_dirs[0] if normal_dirs else None
        if normalized_name in {"mvtec", "mvtec_ad"} and fabric_type:
            return paths["mvtec_ad"] / fabric_type / "train" / "good"
        if normalized_name in {"reference_ready", "processed_reference_ready"}:
            return paths["reference_ready"] / fabric_type if fabric_type else paths["reference_ready"]
        return None

    def count_images(self, path: Path) -> int:
        return _count_files(path, IMAGE_EXTENSIONS)

    def list_images(self, path: Path, max_images: int | None = None) -> list[Path]:
        if not path.exists():
            return []
        images = sorted(
            (
                file_path
                for file_path in path.rglob("*")
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda item: str(item).lower(),
        )
        if max_images is None:
            return images
        return images[:max_images]

    def _discover_company(self, root: Path) -> DatasetSummary:
        normal_dirs = self._company_normal_dirs(root)
        normal_images = {
            image_path
            for normal_dir in normal_dirs
            for image_path in self.list_images(normal_dir)
        }
        normal_count = len(normal_images)
        real_images = {
            image_path
            for image_path in self.list_images(root)
            if "fabric_synthetic_testset" not in {part.lower() for part in image_path.relative_to(root).parts}
        }
        defective_count = len(real_images - normal_images)
        synthetic_good = root / "fabric_synthetic_testset"
        synthetic_normal_count = sum(
            _count_files(path, IMAGE_EXTENSIONS)
            for path in synthetic_good.rglob("synthetic_good")
            if path.is_dir()
        ) if synthetic_good.exists() else 0
        synthetic_defect_count = sum(
            _count_files(path, IMAGE_EXTENSIONS)
            for path in synthetic_good.rglob("synthetic_defect")
            if path.is_dir()
        ) if synthetic_good.exists() else 0
        synthetic_mask_count = sum(
            _count_files(path, MASK_EXTENSIONS)
            for path in synthetic_good.rglob("defect_masks")
            if path.is_dir()
        ) if synthetic_good.exists() else 0
        classes = []
        if normal_count:
            classes.append(
                DatasetClassSummary(
                    name="normal",
                    role="normal_reference",
                    path=_path_string(root),
                    image_count=normal_count,
                )
            )
        if defective_count:
            classes.append(
                DatasetClassSummary(
                    name="defective",
                    role="defective_or_marked_iqc",
                    path=_path_string(root),
                    image_count=defective_count,
                )
            )
        if synthetic_normal_count:
            classes.append(
                DatasetClassSummary(
                    name="synthetic_normal_test",
                    role="normal_test",
                    path=_path_string(synthetic_good),
                    image_count=synthetic_normal_count,
                )
            )
        if synthetic_defect_count:
            classes.append(
                DatasetClassSummary(
                    name="synthetic_defective_test",
                    role="defective_test",
                    path=_path_string(synthetic_good),
                    image_count=synthetic_defect_count,
                )
            )
        if synthetic_mask_count:
            classes.append(
                DatasetClassSummary(
                    name="synthetic_defect_masks",
                    role="mask",
                    path=_path_string(synthetic_good),
                    image_count=0,
                    mask_count=synthetic_mask_count,
                )
            )
        return DatasetSummary(
            name="company_iqc",
            display_name="Company IQC Fabric Images",
            dataset_type="company_defect_samples",
            source_path=_path_string(root),
            exists=root.exists(),
            normal_paths=[_path_string(path) for path in normal_dirs],
            classes=classes,
            notes=(
                "Real good images are reference candidates. Synthetic images are isolated as a stress-test set; "
                "original provenance copies and masks are excluded from model input counts."
            ),
        )

    def _company_normal_dirs(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        return sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_dir() and _is_company_normal_folder(path.name)
            ),
            key=lambda path: str(path).lower(),
        )

    def _discover_lusitano(self, root: Path) -> DatasetSummary:
        normal = root / "nondefects"
        normal_count = _count_files(normal, IMAGE_EXTENSIONS)
        return DatasetSummary(
            name="lusitano",
            display_name="Lusitano Fabric Dataset",
            dataset_type="fabric_normal_reference",
            source_path=_path_string(root),
            exists=root.exists(),
            normal_paths=[_path_string(normal)] if normal.exists() else [],
            classes=[
                DatasetClassSummary(
                    name="nondefects",
                    role="normal_reference",
                    path=_path_string(normal),
                    image_count=normal_count,
                )
            ] if normal.exists() else [],
            notes="All images under nondefects are normal-reference candidates.",
        )

    def _discover_dme(self, root: Path) -> DatasetSummary:
        classes: list[DatasetClassSummary] = []
        normal_paths: list[str] = []
        if root.exists():
            for folder in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
                role = "normal" if folder.name.lower() in {"good", "normal", "non defective", "non_defective"} else "defective"
                if role == "normal":
                    normal_paths.append(_path_string(folder))
                classes.append(
                    DatasetClassSummary(
                        name=folder.name,
                        role=role,
                        path=_path_string(folder),
                        image_count=_count_files(folder, IMAGE_EXTENSIONS),
                    )
                )

        return DatasetSummary(
            name="dme_20250129",
            display_name="DME 20250129 Fabric Defects",
            dataset_type="fabric_classification",
            source_path=_path_string(root),
            exists=root.exists(),
            normal_paths=normal_paths,
            classes=classes,
            notes="Use non defective as the main normal-reference folder; other folders are defect examples.",
        )

    def _discover_wfdd(self, root: Path) -> DatasetSummary:
        classes: list[DatasetClassSummary] = []
        normal_paths: list[str] = []
        if root.exists():
            for cloth_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
                train_good = cloth_dir / "train" / "good"
                if train_good.exists():
                    normal_paths.append(_path_string(train_good))
                    classes.append(self._class_summary(f"{cloth_dir.name}/train/good", "normal_reference", train_good))

                test_dir = cloth_dir / "test"
                if test_dir.exists():
                    for test_class in sorted((item for item in test_dir.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
                        role = "normal_test" if test_class.name.lower() == "good" else "defective_test"
                        classes.append(self._class_summary(f"{cloth_dir.name}/test/{test_class.name}", role, test_class))

                gt_dir = cloth_dir / "ground_truth"
                if gt_dir.exists():
                    for mask_class in sorted((item for item in gt_dir.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
                        classes.append(
                            DatasetClassSummary(
                                name=f"{cloth_dir.name}/ground_truth/{mask_class.name}",
                                role="mask",
                                path=_path_string(mask_class),
                                image_count=0,
                                mask_count=_count_files(mask_class, MASK_EXTENSIONS),
                            )
                        )

        return DatasetSummary(
            name="wfdd",
            display_name="Woven Fabric Defect Detection Dataset",
            dataset_type="fabric_anomaly_localization",
            source_path=_path_string(root),
            exists=root.exists(),
            normal_paths=normal_paths,
            classes=classes,
            notes="Use each cloth train/good folder as a separate reference library for that fabric type.",
        )

    def _discover_mvtec(self, root: Path) -> DatasetSummary:
        classes: list[DatasetClassSummary] = []
        normal_paths: list[str] = []
        if root.exists():
            for category_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
                train_good = category_dir / "train" / "good"
                if train_good.exists():
                    normal_paths.append(_path_string(train_good))
                    classes.append(self._class_summary(f"{category_dir.name}/train/good", "normal_reference", train_good))

                test_dir = category_dir / "test"
                if test_dir.exists():
                    for test_class in sorted((item for item in test_dir.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
                        role = "normal_test" if test_class.name.lower() == "good" else "defective_test"
                        classes.append(self._class_summary(f"{category_dir.name}/test/{test_class.name}", role, test_class))

                gt_dir = category_dir / "ground_truth"
                if gt_dir.exists():
                    for mask_class in sorted((item for item in gt_dir.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
                        classes.append(
                            DatasetClassSummary(
                                name=f"{category_dir.name}/ground_truth/{mask_class.name}",
                                role="mask",
                                path=_path_string(mask_class),
                                image_count=0,
                                mask_count=_count_files(mask_class, MASK_EXTENSIONS),
                            )
                        )

        return DatasetSummary(
            name="mvtec_ad",
            display_name="MVTec AD",
            dataset_type="general_anomaly_validation",
            source_path=_path_string(root),
            exists=root.exists(),
            normal_paths=normal_paths,
            classes=classes,
            notes="Use for early pipeline validation before fabric-specific tuning.",
        )

    def _discover_reference_ready(self, root: Path) -> DatasetSummary:
        classes: list[DatasetClassSummary] = []
        normal_paths: list[str] = []
        if root.exists():
            for library_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
                normal_paths.append(_path_string(library_dir))
                classes.append(self._class_summary(library_dir.name, "normal_reference", library_dir))

            loose_image_count = sum(
                1
                for file_path in root.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if loose_image_count:
                normal_paths.append(_path_string(root))
                classes.append(
                    DatasetClassSummary(
                        name="root",
                        role="normal_reference",
                        path=_path_string(root),
                        image_count=loose_image_count,
                    )
                )

        return DatasetSummary(
            name="reference_ready",
            display_name="Processed Reference-Ready Image Sets",
            dataset_type="prepared_reference_images",
            source_path=_path_string(root),
            exists=root.exists(),
            normal_paths=normal_paths,
            classes=classes,
            notes="Optional curated normal-image input layer for future multi-fabric reference libraries.",
        )

    def _class_summary(self, name: str, role: str, path: Path) -> DatasetClassSummary:
        return DatasetClassSummary(
            name=name,
            role=role,
            path=_path_string(path),
            image_count=_count_files(path, IMAGE_EXTENSIONS),
        )


def _is_company_normal_folder(name: str) -> bool:
    normalized = re.sub(r"\(\d+\)$", "", name.strip().lower())
    normalized = re.sub(r"[\s-]+", "_", normalized).strip("_")
    return normalized in COMPANY_NORMAL_FOLDER_NAMES or normalized.startswith("good_")
