import csv
from pathlib import Path

from peri_magnetics.models.bh_interpolation import BHInterpolator


class MaterialDatabase:

    def __init__(self):
        self.data_dir = Path(__file__).resolve().parents[1] / "data"
        self.database_file = (
            self.data_dir / "materials" / "material_database.csv"
        )

    def load_materials(self):
        """
        재료 목록 파일을 읽어 등록된 재료 정보를 반환한다.
        """

        if not self.database_file.exists():
            raise FileNotFoundError(
                f"재료 목록 파일을 찾을 수 없습니다: {self.database_file}"
            )

        with open(
            self.database_file,
            mode="r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)
            materials = list(reader)

        return materials

    def get_material(self, material_id):
        """
        재료 ID로 재료 정보를 검색한다.
        """

        materials = self.load_materials()

        for material in materials:
            if material["material_id"] == material_id:

                bh_path = (
                    self.data_dir
                    / "materials"
                    / material["bh_data_file"]
                )

                if not bh_path.exists():
                    raise FileNotFoundError(
                        f"B-H 데이터 파일을 찾을 수 없습니다: {bh_path}"
                    )

                material["bh_data_path"] = bh_path

                return material

        raise ValueError(
            f"등록되지 않은 재료입니다: {material_id}"
        )

    def search_by_manufacturer(self, manufacturer):
        """
        제조사 이름으로 등록된 재료를 검색한다.
        대소문자를 구분하지 않는다.
        """

        materials = self.load_materials()

        results = [
            material
            for material in materials
            if material["manufacturer"].strip().casefold()
            == manufacturer.strip().casefold()
        ]

        return results

    def search_by_name(self, material_name):
        """
        재료명으로 등록된 재료를 검색한다.
        대소문자를 구분하지 않는다.
        """

        materials = self.load_materials()

        results = [
            material
            for material in materials
            if material["material_name"].strip().casefold()
            == material_name.strip().casefold()
        ]

        return results

    def search_by_id(self, material_id):
        """
        재료 ID로 등록된 재료를 검색한다.
        """

        materials = self.load_materials()

        results = [
            material
            for material in materials
            if material["material_id"].strip().casefold()
            == material_id.strip().casefold()
        ]

        return results

    def print_materials(self, materials):
        """
        검색된 재료 목록을 표 형태로 출력한다.
        """

        if not materials:
            print("검색된 재료가 없습니다.")
            return

        print("-" * 65)
        print(
            f"{'재료 ID':<18}"
            f"{'제조사':<15}"
            f"{'재료명':<15}"
            f"{'지원 온도':<15}"
        )
        print("-" * 65)

        for material in materials:
            print(
                f"{material['material_id']:<18}"
                f"{material['manufacturer']:<15}"
                f"{material['material_name']:<15}"
                f"{material['available_temperatures_C']:<15}"
            )

        print("-" * 65)

    def select_material(self, material_id):
        """
        재료 ID를 입력받아 해당 재료의 B-H 계산기를 생성한다.
        """

        material = self.get_material(material_id)

        bh_model = self.create_bh_interpolator(material_id)

        print(
            f"선택된 재료: "
            f"{material['manufacturer']} "
            f"{material['material_name']}"
        )

        return bh_model

    def create_bh_interpolator(self, material_id):
        """
        선택한 재료의 B-H 데이터를 불러와 계산기를 생성한다.
        """

        material = self.get_material(material_id)

        bh_model = BHInterpolator(
            csv_path=material["bh_data_path"]
        )

        return bh_model
    
    def validate_material(self, material_id):
        """
        등록된 재료의 B-H 데이터 파일을 검증한다.
        """

        import pandas as pd
        import numpy as np

        material = self.get_material(material_id)
        bh_path = material["bh_data_path"]

        df = pd.read_csv(
            bh_path,
            encoding="cp949"
        )

        temperatures = [
            float(temp)
            for temp in material["available_temperatures_C"].split(";")
        ]

        required_columns = []

        for temp in temperatures:
            temp_label = f"{temp:g}"

            required_columns.extend([
                f"H {temp_label}℃ (A/m)",
                f"B {temp_label}℃ (T)"
            ])

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"필수 컬럼이 없습니다: {missing_columns}"
            )

        if df[required_columns].isnull().any().any():
            raise ValueError("B-H 데이터에 결측값이 있습니다.")

        for column in required_columns:
            values = df[column].to_numpy(dtype=float)

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{column}에 유효하지 않은 숫자가 있습니다."
                )

            if np.any(np.diff(values) < 0):
                raise ValueError(
                    f"{column} 데이터가 단조 증가하지 않습니다."
                )
            if np.any(values < 0):
                raise ValueError(
                    f"{column}에 음수 데이터가 있습니다."
                )
        temperature_ranges = {}

        for temp in temperatures:
            temp_label = f"{temp:g}"

            H_column = f"H {temp_label}℃ (A/m)"
            B_column = f"B {temp_label}℃ (T)"

            temperature_ranges[temp] = {
                "H_min": float(df[H_column].min()),
                "H_max": float(df[H_column].max()),
                "B_min": float(df[B_column].min()),
                "B_max": float(df[B_column].max())
            }

        return {
            "material_id": material_id,
            "material_name": material["material_name"],
            "data_rows": len(df),
            "columns": required_columns,
            "temperature_ranges": temperature_ranges,
            "status": "PASS"
        }