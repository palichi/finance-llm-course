"""
indicators/technical.py 단위 테스트.

실행: pytest tests/test_technical.py -v  (프로젝트 루트에서)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from indicators.technical import compute_indicators, _rsi_wilder


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

def make_df(code: str, prices: list[float]) -> pd.DataFrame:
    """종목코드 하나의 DataFrame 생성 헬퍼 (원본 컬럼명 사용)."""
    n     = len(prices)
    # pd.date_range로 비즈니스 데이를 생성해 YYYYMMDD 정수로 변환
    dates = [int(d.strftime("%Y%m%d"))
             for d in pd.date_range("20240101", periods=n, freq="B")]
    return pd.DataFrame({
        "srtnCd": code,
        "basDt" : dates,
        "mkp"   : prices,
        "clpr"  : prices,
        "hipr"  : prices,
        "lopr"  : prices,
    })


def make_two_stock_df() -> pd.DataFrame:
    """A, B 두 종목을 이어붙인 DataFrame."""
    prices_a = [100.0 + i for i in range(80)]
    prices_b = [500.0 - i for i in range(80)]
    return pd.concat([
        make_df("A", prices_a),
        make_df("B", prices_b),
    ], ignore_index=True)


# ---------------------------------------------------------------------------
# 0. 컬럼 표준화 (신규)
# ---------------------------------------------------------------------------

class TestColumnStandardization:
    def test_ticker_column_exists(self):
        out = compute_indicators(make_df("X", list(range(1, 30))))
        assert "ticker" in out.columns

    def test_original_srtncd_removed(self):
        out = compute_indicators(make_df("X", list(range(1, 30))))
        assert "srtnCd" not in out.columns

    def test_close_column_exists(self):
        out = compute_indicators(make_df("X", list(range(1, 30))))
        assert "close" in out.columns

    def test_original_clpr_removed(self):
        out = compute_indicators(make_df("X", list(range(1, 30))))
        assert "clpr" not in out.columns

    def test_date_column_exists(self):
        out = compute_indicators(make_df("X", list(range(1, 30))))
        assert "date" in out.columns

    def test_original_basdt_removed(self):
        out = compute_indicators(make_df("X", list(range(1, 30))))
        assert "basDt" not in out.columns

    def test_date_is_datetime(self):
        out = compute_indicators(make_df("X", list(range(1, 30))))
        assert pd.api.types.is_datetime64_any_dtype(out["date"])

    def test_date_ascending_within_ticker(self):
        out = compute_indicators(make_df("X", list(range(1, 30))))
        assert out["date"].is_monotonic_increasing

    def test_optional_columns_renamed_if_present(self):
        """itmsNm, trqu 컬럼이 있을 때만 rename."""
        df = make_df("X", list(range(1, 30)))
        df["itmsNm"] = "테스트"
        df["trqu"]   = 1000
        out = compute_indicators(df)
        assert "name"   in out.columns
        assert "volume" in out.columns
        assert "itmsNm" not in out.columns
        assert "trqu"   not in out.columns


# ---------------------------------------------------------------------------
# 1. 기본 컬럼 생성 확인
# ---------------------------------------------------------------------------

class TestOutputColumns:
    def test_all_indicator_columns_present(self):
        df  = make_df("X", list(range(1, 71)))
        out = compute_indicators(df)
        expected = {
            "sma5", "sma20", "sma60", "ema20",
            "golden_flag", "dead_flag",
            "disparity20", "disparity20_centered",
            "rsi14", "rsi14_norm",
        }
        assert expected.issubset(set(out.columns))

    def test_row_count_preserved(self):
        prices = list(range(1, 101))
        df     = make_df("X", prices)
        out    = compute_indicators(df)
        assert len(out) == len(df)


# ---------------------------------------------------------------------------
# 2. SMA / EMA 값 검증
# ---------------------------------------------------------------------------

class TestMovingAverages:
    def test_sma5_first_valid(self):
        out = compute_indicators(make_df("X", [10.0] * 10))
        assert out["sma5"].iloc[4] == pytest.approx(10.0)
        assert pd.isna(out["sma5"].iloc[3])

    def test_sma20_known_value(self):
        out = compute_indicators(make_df("X", list(range(1, 25))))
        # 인덱스 19 (20번째 행): mean(1..20) = 10.5
        assert out["sma20"].iloc[19] == pytest.approx(10.5)

    def test_sma60_nan_before_60th(self):
        out = compute_indicators(make_df("X", list(range(1, 65))))
        assert pd.isna(out["sma60"].iloc[58])
        assert not pd.isna(out["sma60"].iloc[59])

    def test_ema20_constant_price(self):
        """가격이 일정하면 EMA도 같아야 함."""
        out = compute_indicators(make_df("X", [100.0] * 30))
        assert out["ema20"].iloc[25] == pytest.approx(100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. RSI14 검증
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_all_up_equals_100(self):
        out = compute_indicators(make_df("X", [float(i) for i in range(1, 30)]))
        assert out["rsi14"].iloc[25] == pytest.approx(100.0, abs=1e-4)

    def test_rsi_all_down_equals_0(self):
        out = compute_indicators(make_df("X", [float(30 - i) for i in range(30)]))
        assert out["rsi14"].iloc[25] == pytest.approx(0.0, abs=1e-4)

    def test_rsi_range(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100)) + 100
        out = compute_indicators(make_df("X", prices.clip(min=1).tolist())).dropna(subset=["rsi14"])
        assert (out["rsi14"] >= 0).all() and (out["rsi14"] <= 100).all()

    def test_rsi14_norm_range(self):
        np.random.seed(7)
        prices = np.cumsum(np.random.randn(100)) + 100
        out = compute_indicators(make_df("X", prices.clip(min=1).tolist())).dropna(subset=["rsi14_norm"])
        assert (out["rsi14_norm"] >= -1).all() and (out["rsi14_norm"] <= 1).all()

    def test_rsi_known_value(self):
        base   = [float(i) for i in range(1, 16)]
        tail   = [base[-1] - 7]
        rsi    = _rsi_wilder(pd.Series(base + tail), 14)
        assert 50 < rsi.iloc[-1] < 90

    def test_rsi_no_loss_no_division_error(self):
        """평균 손실 0 구간: 첫 행 NaN(diff 특성)은 정상, 이후 모두 100."""
        rsi = _rsi_wilder(pd.Series([float(i) for i in range(1, 20)]), 14)
        assert not rsi.iloc[1:].isna().any()
        assert rsi.iloc[-1] == pytest.approx(100.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. 골든/데드크로스 플래그
# ---------------------------------------------------------------------------

class TestCrossFlags:
    def _cross_df(self, direction: str) -> pd.DataFrame:
        if direction == "golden":
            prices = [10.0] * 30 + [50.0] * 30
        else:
            prices = [50.0] * 30 + [10.0] * 30
        return make_df("X", prices)

    def test_golden_flag_fires(self):
        out = compute_indicators(self._cross_df("golden"), cross_window=5)
        assert out["golden_flag"].iloc[30:40].max() == 1

    def test_dead_flag_fires(self):
        out = compute_indicators(self._cross_df("dead"), cross_window=5)
        assert out["dead_flag"].iloc[30:40].max() == 1

    def test_flags_are_binary(self):
        out = compute_indicators(self._cross_df("golden"))
        assert set(out["golden_flag"].unique()).issubset({0, 1})
        assert set(out["dead_flag"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# 5. 이격도
# ---------------------------------------------------------------------------

class TestDisparity:
    def test_disparity20_formula(self):
        out = compute_indicators(make_df("X", [100.0] * 25))
        assert out["disparity20"].iloc[24] == pytest.approx(100.0)

    def test_disparity20_centered_zero(self):
        out = compute_indicators(make_df("X", [100.0] * 25))
        assert out["disparity20_centered"].iloc[24] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. 종목 경계 격리 (핵심 검증)
# ---------------------------------------------------------------------------

class TestStockBoundaryIsolation:
    def test_sma_not_contaminated_across_stocks(self):
        """
        A종목 마지막 값(179)과 B종목 첫 값(500)이 섞이면
        B종목 초기 SMA5가 A 데이터의 영향을 받아 잘못된 값이 나온다.
        종목별 독립 계산이면 B의 SMA5[4] = mean(500,499,498,497,496) = 498.
        """
        out    = compute_indicators(make_two_stock_df())
        b_rows = out[out["ticker"] == "B"].reset_index(drop=True)
        assert b_rows["sma5"].iloc[4] == pytest.approx(
            np.mean([500, 499, 498, 497, 496]), rel=1e-6
        )

    def test_rsi_not_contaminated_across_stocks(self):
        df = pd.concat([
            make_df("A", [float(i) for i in range(1, 81)]),
            make_df("B", [float(80 - i) for i in range(80)]),
        ], ignore_index=True)
        out    = compute_indicators(df)
        a_rows = out[out["ticker"] == "A"].dropna(subset=["rsi14"]).reset_index(drop=True)
        b_rows = out[out["ticker"] == "B"].dropna(subset=["rsi14"]).reset_index(drop=True)
        assert a_rows["rsi14"].iloc[-1] > 90
        assert b_rows["rsi14"].iloc[-1] < 10

    def test_ema_boundary_isolation(self):
        df = pd.concat([
            make_df("A", [1000.0] * 80),
            make_df("B", [1.0]    * 80),
        ], ignore_index=True)
        out    = compute_indicators(df)
        b_rows = out[out["ticker"] == "B"].reset_index(drop=True)
        assert b_rows["ema20"].iloc[30] == pytest.approx(1.0, rel=1e-3)

    def test_golden_flag_boundary_isolation(self):
        df = pd.concat([
            make_df("A", [10.0] * 30 + [100.0] * 50),
            make_df("B", [50.0] * 80),
        ], ignore_index=True)
        out    = compute_indicators(df)
        b_rows = out[out["ticker"] == "B"].dropna(subset=["sma5", "sma20"])
        assert b_rows["golden_flag"].max() == 0


# ---------------------------------------------------------------------------
# 7. nan_policy 옵션
# ---------------------------------------------------------------------------

class TestNanPolicy:
    def test_keep_has_nans(self):
        out = compute_indicators(make_df("X", list(range(1, 70))), nan_policy="keep")
        assert out["sma60"].isna().any()

    def test_drop_removes_nan_rows(self):
        out = compute_indicators(make_df("X", list(range(1, 70))), nan_policy="drop")
        assert not out["sma60"].isna().any()
        assert not out["rsi14"].isna().any()

    def test_backfill_no_nans(self):
        out = compute_indicators(make_df("X", list(range(1, 70))), nan_policy="backfill")
        cols = ["sma5", "sma20", "sma60", "ema20",
                "disparity20", "disparity20_centered", "rsi14", "rsi14_norm"]
        assert not out[cols].isna().any().any()

    def test_drop_multi_stock_row_count(self):
        """drop: 두 종목 모두 앞 구간 제거 → 각 80행 중 sma60 유효=21행."""
        out = compute_indicators(make_two_stock_df(), nan_policy="drop")
        assert len(out) == 2 * (80 - 59)
