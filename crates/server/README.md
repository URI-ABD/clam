# CLAM Web Server

The CLAM web server is a RESTful web server that provides a simple interface to the CLAM library.
It is built using the [Rocket](https://rocket.rs/) web framework.

## TODO

- [ ] OpenAPI Documentation (Blocked on https://github.com/GREsau/okapi/pull/149)

## Usage

You can run this with fake data, using `CLAM_BOOTSTRAP=1 cargo run --bin server`.
Or, you can run with your own data, using `CLAM_CODEC_DATA_PATH=/path/to/your/data CLAM_SQUISHY_BALL_PATH=/path/to/your/data cargo run --bin server`.
