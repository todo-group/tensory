{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nix-jyjyjcr = {
      url = "github:jyjyjcr/nix-jyjyjcr";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      rust-overlay,
      nix-jyjyjcr,
      ...
    }:
    (flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfreePredicate =
            pkg:
            builtins.elem (pkgs.lib.getName pkg) [
              "corefonts"
            ];
        };
        tensory-logo = pkgs.callPackage ./assets/logo.nix { };

        pkgs-dev = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
            nix-jyjyjcr.overlays.default
          ];
        };
      in
      {
        packages.tensory-logo = tensory-logo;

        devShells = pkgs-dev.alt-shell.mkCommonShells { } {
          packages = [
            (pkgs-dev.rust-bin.stable.latest.default.override {
              extensions = [ "rust-src" ];
            })
            pkgs-dev.cargo-edit
            # for openblas
            pkgs-dev.openblas
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

            pkgs-dev.pkg-config
            pkgs-dev.uv
            pkgs-dev.python313
            pkgs-dev.sccache
          ];
        };
      }
    ))
    // {
      overlays.default = final: prev: {
        tensory-logo = prev.callPackage ./assets/logo.nix { };
      };
    };
}
