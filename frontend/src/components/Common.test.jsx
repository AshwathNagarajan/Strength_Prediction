import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Metric, Notice } from "./Common";

describe("shared engineering UI", () => {
  it("renders model metrics without replacing zero values", () => {
    render(<Metric label="RMSE" value={0} hint="kN" />);
    expect(screen.getByText("RMSE")).toBeTruthy();
    expect(screen.getByText("0")).toBeTruthy();
    expect(screen.getByText("kN")).toBeTruthy();
  });

  it("exposes warning content", () => {
    render(<Notice type="warning">Outside training range</Notice>);
    expect(screen.getByText("Outside training range").className).toContain("warning");
  });
});
