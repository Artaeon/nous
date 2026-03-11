FROM golang:1.22-alpine AS builder

WORKDIR /build
COPY go.mod ./
COPY cmd/ cmd/
COPY internal/ internal/

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o nous ./cmd/nous

FROM alpine:3.19
RUN apk --no-cache add ca-certificates

WORKDIR /app
COPY --from=builder /build/nous .

EXPOSE 3333

# Default: server mode on port 3333
ENTRYPOINT ["./nous"]
CMD ["--serve", "--port", "3333", "--host", "http://host.docker.internal:11434"]
