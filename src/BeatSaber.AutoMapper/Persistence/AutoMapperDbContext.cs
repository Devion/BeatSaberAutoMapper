using Microsoft.EntityFrameworkCore;

namespace BeatSaber.AutoMapper.Persistence;

public sealed class AutoMapperDbContext : DbContext
{
    public DbSet<SongEntity> Songs { get; set; } = null!;
    public DbSet<AudioAnalysisEntity> AudioAnalyses { get; set; } = null!;
    public DbSet<BeatmapSetEntity> BeatmapSets { get; set; } = null!;
    public DbSet<DifficultyEntity> Difficulties { get; set; } = null!;
    public DbSet<ValidationRunEntity> ValidationRuns { get; set; } = null!;
    public DbSet<ValidationIssueEntity> ValidationIssues { get; set; } = null!;
    public DbSet<DatasetManifestEntity> DatasetManifests { get; set; } = null!;
    public DbSet<TrainingRunEntity> TrainingRuns { get; set; } = null!;
    public DbSet<ModelArtifactEntity> ModelArtifacts { get; set; } = null!;
    public DbSet<GenerationRunEntity> GenerationRuns { get; set; } = null!;
    public DbSet<CorpusDuplicateEntity> CorpusDuplicates { get; set; } = null!;

    public AutoMapperDbContext(DbContextOptions<AutoMapperDbContext> options) : base(options) { }

    public static AutoMapperDbContext Create(string dbPath)
    {
        var options = new DbContextOptionsBuilder<AutoMapperDbContext>()
            .UseSqlite($"Data Source={dbPath}")
            .Options;
        return new AutoMapperDbContext(options);
    }

    public void EnsureCreated() => Database.EnsureCreated();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        // Songs
        modelBuilder.Entity<SongEntity>(e =>
        {
            e.HasKey(s => s.Id);
            e.Property(s => s.Title).IsRequired();
            e.HasMany(s => s.BeatmapSets).WithOne(b => b.Song).HasForeignKey(b => b.SongId);
            e.HasOne(s => s.AudioAnalysis).WithOne(a => a.Song).HasForeignKey<AudioAnalysisEntity>(a => a.SongId);
        });

        // AudioAnalysis
        modelBuilder.Entity<AudioAnalysisEntity>(e => e.HasKey(a => a.Id));

        // BeatmapSets
        modelBuilder.Entity<BeatmapSetEntity>(e =>
        {
            e.HasKey(b => b.Id);
            e.HasMany(b => b.Difficulties).WithOne(d => d.BeatmapSet).HasForeignKey(d => d.BeatmapSetId);
        });

        // Difficulties
        modelBuilder.Entity<DifficultyEntity>(e =>
        {
            e.HasKey(d => d.Id);
            e.HasMany(d => d.ValidationRuns).WithOne(v => v.Difficulty).HasForeignKey(v => v.DifficultyId);
        });

        // ValidationRuns
        modelBuilder.Entity<ValidationRunEntity>(e =>
        {
            e.HasKey(v => v.Id);
            e.HasMany(v => v.Issues).WithOne(i => i.ValidationRun).HasForeignKey(i => i.ValidationRunId);
        });

        // ValidationIssues
        modelBuilder.Entity<ValidationIssueEntity>(e => e.HasKey(i => i.Id));

        // DatasetManifests
        modelBuilder.Entity<DatasetManifestEntity>(e => e.HasKey(d => d.Id));

        // TrainingRuns
        modelBuilder.Entity<TrainingRunEntity>(e =>
        {
            e.HasKey(t => t.Id);
            e.HasMany(t => t.ModelArtifacts).WithOne(m => m.TrainingRun).HasForeignKey(m => m.TrainingRunId);
        });

        // ModelArtifacts
        modelBuilder.Entity<ModelArtifactEntity>(e => e.HasKey(m => m.Id));

        // GenerationRuns
        modelBuilder.Entity<GenerationRunEntity>(e =>
        {
            e.HasKey(g => g.Id);
            e.HasMany(g => g.ValidationRuns).WithOne(v => v.GenerationRun).HasForeignKey(v => v.GenerationRunId);
        });

        // CorpusDuplicates
        modelBuilder.Entity<CorpusDuplicateEntity>(e => e.HasKey(c => c.Id));
    }
}
