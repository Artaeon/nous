FROM golang:1.27.3-alpine@sha256:d200d2dd76b03a99b52651fd1325aff7dcd35422c35c01bac56571b0507f3deb AS builder

WORKDIR /build
COPY go.mod ./
COPY cmd/ cmd/
COPY internal/ internal/

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o nous ./cmd/nous

FROM alpine:3.24@sha256:be2656061b536c1883c4034790e21d0a09c5aaaf51968b89368a3c0371ea2c9a
RUN apk --no-cache add ca-certificates git

WORKDIR /app
COPY --from=builder /build/nous .
COPY packages/ /app/packages/

VOLUME ["/data", "/workspace"]

EXPOSE 3333

ENTRYPOINT ["./nous"]
CMD ["--serve", "--public", "--port", "3333", "--memory", "/data", "--trust"]
