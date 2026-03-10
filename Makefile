.PHONY: build run clean test

BINARY := nous
SRC := ./cmd/nous

build:
	go build -o $(BINARY) $(SRC)

run: build
	./$(BINARY)

run-shell: build
	./$(BINARY) --allow-shell

clean:
	rm -f $(BINARY)

test:
	go test ./...

fmt:
	go fmt ./...

vet:
	go vet ./...

lint: fmt vet

# Build for all platforms
release:
	GOOS=linux GOARCH=amd64 go build -o $(BINARY)-linux-amd64 $(SRC)
	GOOS=linux GOARCH=arm64 go build -o $(BINARY)-linux-arm64 $(SRC)
	GOOS=darwin GOARCH=amd64 go build -o $(BINARY)-darwin-amd64 $(SRC)
	GOOS=darwin GOARCH=arm64 go build -o $(BINARY)-darwin-arm64 $(SRC)
