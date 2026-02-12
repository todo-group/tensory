{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nix-jyjyjcr = {
      url = "github:jyjyjcr/nix-jyjyjcr/dev/alt-shell";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { nixpkgs, flake-utils, rust-overlay, nix-jyjyjcr, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
            nix-jyjyjcr.overlays.${system}.default
          ];
        };
      in {
        devShells = pkgs.alt-shell.mkCommonShells { } {
          packages = [
            (pkgs.rust-bin.stable.latest.default.override {
              extensions = [ "rust-src" ];
            })
            pkgs.cargo-edit
            # for openblas
            pkgs.openblas
            # for openblas 64-bit
            # (pkgs.openblas.override { blas64 = true; })

            # for blis (not checked yet)
            # pkgs.blis

            # for netlib
            # pkgs.gfortran
            # pkgs.gfortran.cc
            # pkgs.blas-reference
            # pkgs.lapack-reference

            # for R
            # pkgs.R
            # pkgs.libintl

            pkgs.pkg-config
            pkgs.uv
            pkgs.python313
          ];
        };
      });
}
