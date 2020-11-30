#pragma once
#ifndef __ZIPCONTAINER_H__
#define __ZIPCONTAINER_H__
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#define LOCAL_HEADER_LENGTH 30  /* This is the length w/o file_name and extra_field */
#define LOCAL_HEADER_SIGNATURE 0x04034b50
typedef struct _local_file_header_t
{
    uint32_t  local_file_header_signature;    /*  4 bytes  (LOCAL_HEADER_SIGNATURE) */
    uint16_t  version_needed_to_extract;      /*  2 bytes */
    uint16_t  general_purpose_bit_flag;       /*  2 bytes */
    uint16_t  compression_method;             /*  2 bytes */
    uint16_t  last_mod_file_time;             /*  2 bytes */
    uint16_t  last_mod_file_date;             /*  2 bytes */
    uint32_t  crc_32;                         /*  4 bytes */
    uint32_t  compressed_size;                /*  4 bytes */
    uint32_t  uncompressed_size;              /*  4 bytes */
    uint16_t  file_name_length;               /*  2 bytes */
    uint16_t  extra_field_length;             /*  2 bytes */
    char* file_name;                     /*  (variable size) */
    char* extra_field;                   /*  (variable size) */
} local_file_header_t;

#if 0
/* FIXME. This structure is not yet in use in this library. */
#define DATA_DESCRIPTOR_LENGTH 12
#define DATA_DESCRIPTOR_SIGNATURE 0x08074b50 
typedef struct _data_descriptor_t
{
    uint32_t signature;
    uint32_t crc_32;                          /*  4 bytes */
    uint32_t compressed_size;                 /*  4 bytes */
    uint32_t uncompressed_size;               /*  4 bytes */
} data_descriptor_t;
#endif

#define CENTRAL_DIRECTORY_HEADER_LENGTH 46
#define CENTRAL_DIRECTORY_HEADER_SIGNATURE 0x02014b50
typedef struct _central_directory_header_t
{
    uint32_t central_file_header_signature;   /* 4 bytes  (CENTRAL_DIRECTORY_HEADER_SIGNATURE) */
    uint16_t version_made_by;                 /* 2 bytes */
    uint16_t version_needed_to_extract;       /* 2 bytes */
    uint16_t general_purpose_bit_flag;        /* 2 bytes */
    uint16_t compression_method;              /* 2 bytes */
    uint16_t last_mod_file_time;              /* 2 bytes */
    uint16_t last_mod_file_date;              /* 2 bytes */
    uint32_t crc_32;                          /* 4 bytes */
    uint32_t compressed_size;                 /* 4 bytes */
    uint32_t uncompressed_size;               /* 4 bytes */
    uint16_t file_name_length;                /* 2 bytes */
    uint16_t extra_field_length;              /* 2 bytes */
    uint16_t file_comment_length;             /* 2 bytes */
    uint16_t disk_number_start;               /* 2 bytes */
    uint16_t internal_file_attributes;        /* 2 bytes */
    uint32_t external_file_attributes;        /* 4 bytes */
    uint32_t relative_offset_of_local_header; /* 4 bytes */

    char* file_name;                       /*  (variable size) */
    char* extra_field;                     /*  (variable size) */
    char* file_comment;                    /*  (variable size) */
} central_directory_header_t;

#if 0
/* This is actually not used yet in this library */
#define DIGITAL_SIGNATURE_LENGTH 6
#define DIGITAL_SIGNATURE_SIGNATURE 0x05054b50
typedef struct _digital_signature_t
{
    uint32_t  header_signature;               /*  4 bytes  (0x05054b50) */
    uint16_t  size_of_data;                   /*  2 bytes */
    char* signature_data;                 /*  (variable size) */
} digital_signature_t;
#endif

#define END_OF_CENTRAL_DIR_LENGTH 22
#define END_OF_CENTRAL_DIR_SIGNATURE 0x06054b50
typedef struct _end_of_central_dir_t
{
    uint32_t end_of_central_dir_signature;    /*  4 bytes (END_OF_CENTRAL_DIR_SIGNATURE) */
    uint16_t number_of_this_disk;             /*  2 bytes */
    uint16_t number_of_the_disk_start_of_cd;  /*  2 bytes */
    uint16_t total_num_entries_this_disk;     /*  2 bytes */
    uint16_t total_num_entries_cd;            /*  2 bytes */
    uint32_t size_of_cd;                      /*  4 bytes */
    uint32_t offset_cd_wrt_disknum;           /*  4 bytes */
    uint16_t ZIP_file_comment_length;         /*  2 bytes */
    char* ZIP_file_comment;                /*  (variable size) */
} end_of_central_dir_t;


local_file_header_t* local_file_header_new_from_fp(FILE* fp);
central_directory_header_t* central_directory_header_new_from_fp(FILE* fp);
end_of_central_dir_t* end_of_central_dir_new_from_fp(FILE* fp);

void          local_file_header_free(local_file_header_t* p);
void   central_directory_header_free(central_directory_header_t* p);
void         end_of_central_dir_free(end_of_central_dir_t* p);

/* FIXME: Some of these functions should return a code (int) if things go wrong */
void _write_local_fileheader(FILE* fp, const local_file_header_t* lfh);
void _write_cental_directory_fileheader(FILE* fp, const central_directory_header_t* cdh);
void _write_end_of_central_dir(FILE* fp, const end_of_central_dir_t* eocd);

#ifndef NDEBUG
void _dump_local_fileheader(const local_file_header_t* lfh);
void _dump_central_directory_fileheader(const central_directory_header_t* cdh);
void _dump_end_of_central_dir(const end_of_central_dir_t* eocd);
#endif
#endif /* __ZIPCONTAINER_H__*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

local_file_header_t* local_file_header_new();
central_directory_header_t* central_directory_header_new();
end_of_central_dir_t* end_of_central_dir_new();

/* Argh! I cannot make this macro work properly!! Anyone got an idea? */
#define READ_VARIABLE_LEN_FIELD(datastruct,field) \
    do { \
        size_t len = datastruct ##->## field ## _length; \
        char *buf = malloc( (len + 1) * sizeof(char));\
        size_t chk = fread( buf, len, sizeof(char), fp );\
        if(chk != len) fprintf(stderr, "Incorrect read.\n");\
        datastruct ## -> ## field = buf; \
    } while(0); 

local_file_header_t* local_file_header_new_from_fp(FILE* fp)
{
    local_file_header_t* lfh = local_file_header_new();

    char header[LOCAL_HEADER_LENGTH];
    size_t chk = fread(header, 1, LOCAL_HEADER_LENGTH, fp); /* FIXME */
    if (chk != LOCAL_HEADER_LENGTH) {
        fprintf(stderr, "Cannot read header.\n");
        local_file_header_free(lfh);
        return NULL;
    }
    if (*(uint32_t*)(header) != LOCAL_HEADER_SIGNATURE) {
        fseek(fp, -LOCAL_HEADER_LENGTH, SEEK_CUR);
        local_file_header_free(lfh);
        return NULL;
    }

    /* We cannot assume the structure is "packed" we therefore assign one and one element */
    lfh->local_file_header_signature = *(uint32_t*)(header);      /*  4 bytes */
    lfh->version_needed_to_extract = *(uint16_t*)(header + 4);    /*  2 bytes */
    lfh->general_purpose_bit_flag = *(uint16_t*)(header + 6);    /*  2 bytes */
    lfh->compression_method = *(uint16_t*)(header + 8);    /*  2 bytes */
    lfh->last_mod_file_time = *(uint16_t*)(header + 10);   /*  2 bytes */
    lfh->last_mod_file_date = *(uint16_t*)(header + 12);   /*  2 bytes */
    lfh->crc_32 = *(uint32_t*)(header + 14);   /*  4 bytes */
    lfh->compressed_size = *(uint32_t*)(header + 18);   /*  4 bytes */
    lfh->uncompressed_size = *(uint32_t*)(header + 22);   /*  4 bytes */
    lfh->file_name_length = *(uint16_t*)(header + 26);   /*  2 bytes */
    lfh->extra_field_length = *(uint16_t*)(header + 28);   /*  2 bytes */

    /* Argh! The macro doesn't work */
    do {
        size_t len = lfh->file_name_length;
        char* buf = malloc((len + 1) * sizeof(char));
        size_t chk = fread(buf, sizeof(char), len, fp);
        if (chk != len) fprintf(stderr, "%d != %d.\n", (int)chk, (int)len);
        lfh->file_name = buf;
    } while (0);
    do {
        size_t len = lfh->extra_field_length;
        char* buf = malloc((len + 1) * sizeof(char));
        size_t chk = fread(buf, sizeof(char), len, fp);
        if (chk != len) fprintf(stderr, "Incorrect read.\n");
        lfh->extra_field = buf;
    } while (0);

    return lfh;
}

void  local_file_header_free(local_file_header_t* p)
{
    if (!p) return;
    if (p->file_name) free(p->file_name);
    if (p->extra_field) free(p->extra_field);
    free(p);
}

local_file_header_t* local_file_header_new()
{
    local_file_header_t* lfh = malloc(sizeof(local_file_header_t));
    assert(lfh);

    lfh->file_name = NULL;
    lfh->extra_field = NULL;
    return lfh;
}



central_directory_header_t* central_directory_header_new_from_fp(FILE* fp)
{
    central_directory_header_t* cdh = central_directory_header_new();
    /* FIXME: Do checks all reads and malloc. */
    char buffer[CENTRAL_DIRECTORY_HEADER_LENGTH];
    size_t chk = fread(buffer, 1, CENTRAL_DIRECTORY_HEADER_LENGTH, fp); /* FIXME */
    if (chk != CENTRAL_DIRECTORY_HEADER_LENGTH) {
        fprintf(stderr, "Cannot read buffer.\n");
        central_directory_header_free(cdh);
        return NULL;
    }

    if (*(uint32_t*)(buffer) != CENTRAL_DIRECTORY_HEADER_SIGNATURE) { /* Whoops! */
        central_directory_header_free(cdh);
        return NULL;   /* Discussion: Maybe we should rewind the filepointer, just to be nice? */
    }

    cdh->central_file_header_signature = *(uint32_t*)(buffer);     /*  4 bytes */
    cdh->version_made_by = *(uint32_t*)(buffer + 4);   /*  2 bytes */
    cdh->version_needed_to_extract = *(uint32_t*)(buffer + 6);   /*  2 bytes */
    cdh->general_purpose_bit_flag = *(uint32_t*)(buffer + 8);   /*  2 bytes */
    cdh->compression_method = *(uint32_t*)(buffer + 10);  /*  2 bytes */
    cdh->last_mod_file_time = *(uint32_t*)(buffer + 12);  /*  2 bytes */
    cdh->last_mod_file_date = *(uint32_t*)(buffer + 14);  /*  2 bytes */
    cdh->crc_32 = *(uint32_t*)(buffer + 16);  /*  4 bytes */
    cdh->compressed_size = *(uint32_t*)(buffer + 20);  /*  4 bytes */
    cdh->uncompressed_size = *(uint32_t*)(buffer + 24);  /*  4 bytes */
    cdh->file_name_length = *(uint32_t*)(buffer + 28);  /*  2 bytes */
    cdh->extra_field_length = *(uint32_t*)(buffer + 30);  /*  2 bytes */
    cdh->file_comment_length = *(uint32_t*)(buffer + 32);  /*  2 bytes */
    cdh->disk_number_start = *(uint32_t*)(buffer + 34);  /*  2 bytes */
    cdh->internal_file_attributes = *(uint32_t*)(buffer + 36);  /*  2 bytes */
    cdh->external_file_attributes = *(uint32_t*)(buffer + 38);  /*  4 bytes */
    cdh->relative_offset_of_local_header = *(uint32_t*)(buffer + 42);  /*  4 bytes */

    /* Argh! The macro doesn't work */
    do {
        size_t len = cdh->file_name_length;
        char* buf = malloc((len + 1) * sizeof(char));
        size_t chk = fread(buf, sizeof(char), len, fp);
        if (chk != len) fprintf(stderr, "Incorrect read.\n");
        cdh->file_name = buf;
    } while (0);
    do {
        size_t len = cdh->extra_field_length;
        char* buf = malloc((len + 1) * sizeof(char));
        size_t chk = fread(buf, sizeof(char), len, fp);
        if (chk != len) fprintf(stderr, "Incorrect read.\n");
        cdh->extra_field = buf;
    } while (0);
    do {
        size_t len = cdh->file_comment_length;
        char* buf = malloc((len + 1) * sizeof(char));
        size_t chk = fread(buf, sizeof(char), len, fp);
        if (chk != len) fprintf(stderr, "Incorrect read.\n");
        cdh->file_comment = buf;
    } while (0);

    return cdh;
}

void  central_directory_header_free(central_directory_header_t* p)
{
    if (!p) return;
    if (p->file_name) free(p->file_name);
    if (p->extra_field) free(p->extra_field);
    if (p->file_comment) free(p->file_comment);
    free(p);
}

central_directory_header_t* central_directory_header_new()
{
    central_directory_header_t* cdh = malloc(sizeof(central_directory_header_t));
    assert(cdh);

    cdh->file_name = NULL;
    cdh->extra_field = NULL;
    cdh->file_comment = NULL;
    return cdh;
}

end_of_central_dir_t* end_of_central_dir_new_from_fp(FILE* fp)
{
    end_of_central_dir_t* eocd = end_of_central_dir_new();
    /* FIXME: Do checks all reads and malloc. */
    char buffer[END_OF_CENTRAL_DIR_LENGTH];
    size_t chk = fread(buffer, 1, END_OF_CENTRAL_DIR_LENGTH, fp); /* FIXME */
    if (chk != END_OF_CENTRAL_DIR_LENGTH) {
        fprintf(stderr, "Cannot read buffer.\n");
        end_of_central_dir_free(eocd);
        return NULL;
    }

    if (*(uint32_t*)(buffer) != END_OF_CENTRAL_DIR_SIGNATURE) { /* Whoops! */
        end_of_central_dir_free(eocd);
        return NULL;
    }

    eocd->end_of_central_dir_signature = *(uint32_t*)(buffer);     /*  4 bytes (END_OF_CENTRAL_DIR_SIGNATURE) */
    eocd->number_of_this_disk = *(uint16_t*)(buffer + 4);   /*  2 bytes */
    eocd->number_of_the_disk_start_of_cd = *(uint16_t*)(buffer + 6);   /*  2 bytes */
    eocd->total_num_entries_this_disk = *(uint16_t*)(buffer + 8);    /*  2 bytes */
    eocd->total_num_entries_cd = *(uint16_t*)(buffer + 10);  /*  2 bytes */
    eocd->size_of_cd = *(uint32_t*)(buffer + 12);  /*  4 bytes */
    eocd->offset_cd_wrt_disknum = *(uint32_t*)(buffer + 16);  /*  4 bytes */
    eocd->ZIP_file_comment_length = *(uint16_t*)(buffer + 20);  /*  2 bytes */

    do {
        size_t len = eocd->ZIP_file_comment_length;
        char* buf = malloc((len + 1) * sizeof(char));
        size_t chk = fread(buf, sizeof(char), len, fp);
        if (chk != len) fprintf(stderr, "Incorrect read.\n");
        eocd->ZIP_file_comment = buf;
    } while (0);
    return eocd;
}

void  end_of_central_dir_free(end_of_central_dir_t* p)
{
    if (!p) return;
    if (p->ZIP_file_comment) free(p->ZIP_file_comment);
    free(p);
}

end_of_central_dir_t* end_of_central_dir_new()
{
    end_of_central_dir_t* eocd = malloc(sizeof(end_of_central_dir_t));
    assert(eocd);

    eocd->ZIP_file_comment = NULL;
    return eocd;
}

void _write_local_fileheader(FILE* fp, const local_file_header_t* lf)
{
    /* FIXME: Do some checks before we call this production quality */
    fwrite(&lf->local_file_header_signature, sizeof(uint32_t), 1, fp);    /*  4 bytes */
    fwrite(&lf->version_needed_to_extract, sizeof(uint16_t), 1, fp);    /*  2 bytes */
    fwrite(&lf->general_purpose_bit_flag, sizeof(uint16_t), 1, fp);    /*  2 bytes */
    fwrite(&lf->compression_method, sizeof(uint16_t), 1, fp);    /*  2 bytes */
    fwrite(&lf->last_mod_file_time, sizeof(uint16_t), 1, fp);    /*  2 bytes */
    fwrite(&lf->last_mod_file_date, sizeof(uint16_t), 1, fp);    /*  2 bytes */
    fwrite(&lf->crc_32, sizeof(uint32_t), 1, fp);    /*  4 bytes */
    fwrite(&lf->compressed_size, sizeof(uint32_t), 1, fp);    /*  4 bytes */
    fwrite(&lf->uncompressed_size, sizeof(uint32_t), 1, fp);    /*  4 bytes */
    fwrite(&lf->file_name_length, sizeof(uint16_t), 1, fp);    /*  2 bytes */
    fwrite(&lf->extra_field_length, sizeof(uint16_t), 1, fp);    /*  2 bytes */
    fwrite(lf->file_name, sizeof(char), lf->file_name_length, fp);
    fwrite(lf->extra_field, sizeof(char), lf->extra_field_length, fp);
}

void _write_cental_directory_fileheader(FILE* fp, const central_directory_header_t* cdh)
{
    /* FIXME: Do some checks before we call this production quality */
    fwrite(&cdh->central_file_header_signature, sizeof(uint32_t), 1, fp);   /* 4 bytes */
    fwrite(&cdh->version_made_by, sizeof(uint16_t), 1, fp);                 /* 2 bytes */
    fwrite(&cdh->version_needed_to_extract, sizeof(uint16_t), 1, fp);       /* 2 bytes */
    fwrite(&cdh->general_purpose_bit_flag, sizeof(uint16_t), 1, fp);        /* 2 bytes */
    fwrite(&cdh->compression_method, sizeof(uint16_t), 1, fp);              /* 2 bytes */
    fwrite(&cdh->last_mod_file_time, sizeof(uint16_t), 1, fp);              /* 2 bytes */
    fwrite(&cdh->last_mod_file_date, sizeof(uint16_t), 1, fp);              /* 2 bytes */
    fwrite(&cdh->crc_32, sizeof(uint32_t), 1, fp);                          /* 4 bytes */
    fwrite(&cdh->compressed_size, sizeof(uint32_t), 1, fp);                 /* 4 bytes */
    fwrite(&cdh->uncompressed_size, sizeof(uint32_t), 1, fp);               /* 4 bytes */
    fwrite(&cdh->file_name_length, sizeof(uint16_t), 1, fp);                /* 2 bytes */
    fwrite(&cdh->extra_field_length, sizeof(uint16_t), 1, fp);              /* 2 bytes */
    fwrite(&cdh->file_comment_length, sizeof(uint16_t), 1, fp);             /* 2 bytes */
    fwrite(&cdh->disk_number_start, sizeof(uint16_t), 1, fp);               /* 2 bytes */
    fwrite(&cdh->internal_file_attributes, sizeof(uint16_t), 1, fp);        /* 2 bytes */
    fwrite(&cdh->external_file_attributes, sizeof(uint32_t), 1, fp);        /* 4 bytes */
    fwrite(&cdh->relative_offset_of_local_header, sizeof(uint32_t), 1, fp); /* 4 bytes */

    fwrite(cdh->file_name, sizeof(char), cdh->file_name_length, fp);    /*  (variable size) */
    fwrite(cdh->extra_field, sizeof(char), cdh->extra_field_length, fp);  /*  (variable size) */
    fwrite(cdh->file_comment, sizeof(char), cdh->file_comment_length, fp); /*  (variable size) */
}

void _write_end_of_central_dir(FILE* fp, const end_of_central_dir_t* eocd)
{
    /* FIXME: Do some checks before we call this production quality */
    fwrite(&eocd->end_of_central_dir_signature, sizeof(uint32_t), 1, fp);    /*  4 bytes (END_OF_CENTRAL_DIR_SIGNATURE) */
    fwrite(&eocd->number_of_this_disk, sizeof(uint16_t), 1, fp);             /*  2 bytes */
    fwrite(&eocd->number_of_the_disk_start_of_cd, sizeof(uint16_t), 1, fp);  /*  2 bytes */
    fwrite(&eocd->total_num_entries_this_disk, sizeof(uint16_t), 1, fp);     /*  2 bytes */
    fwrite(&eocd->total_num_entries_cd, sizeof(uint16_t), 1, fp);            /*  2 bytes */
    fwrite(&eocd->size_of_cd, sizeof(uint32_t), 1, fp);                      /*  4 bytes */
    fwrite(&eocd->offset_cd_wrt_disknum, sizeof(uint32_t), 1, fp);           /*  4 bytes */
    fwrite(&eocd->ZIP_file_comment_length, sizeof(uint16_t), 1, fp);         /*  2 bytes */
    fwrite(eocd->ZIP_file_comment, sizeof(char), eocd->ZIP_file_comment_length, fp);  /*  (variable size) */
}

/* These functions are more or less for my debugging purposes */
#ifndef NDEBUG
void _dump_end_of_central_dir(const end_of_central_dir_t* eocd)
{
    printf("(%d) end_of_central_dir_signature\n", eocd->end_of_central_dir_signature);
    printf("(%d) number_of_this_disk\n", eocd->number_of_this_disk);
    printf("(%d) number_of_the_disk_start_of_cd\n", eocd->number_of_the_disk_start_of_cd);
    printf("(%d) total_num_entries_this_disk\n", eocd->total_num_entries_this_disk);
    printf("(%d) total_num_entries_cd\n", eocd->total_num_entries_cd);
    printf("(%d) size_of_cd\n", eocd->size_of_cd);
    printf("(%d) offset_cd_wrt_disknum\n", eocd->offset_cd_wrt_disknum);
    printf("(%d) ZIP_file_comment_length\n", eocd->ZIP_file_comment_length);
}

void _dump_central_directory_fileheader(const central_directory_header_t* cdh)
{
    printf("(%d) central_file_header_signature\n", cdh->central_file_header_signature);   /* 4 bytes */
    printf("(%d) version_made_by\n", cdh->version_made_by);                 /* 2 bytes */
    printf("(%d) version_needed_to_extract\n", cdh->version_needed_to_extract);       /* 2 bytes */
    printf("(%d) general_purpose_bit_flag\n", cdh->general_purpose_bit_flag);        /* 2 bytes */
    printf("(%d) compression_method\n", cdh->compression_method);              /* 2 bytes */
    printf("(%d) last_mod_file_time\n", cdh->last_mod_file_time);              /* 2 bytes */
    printf("(%d) last_mod_file_date\n", cdh->last_mod_file_date);              /* 2 bytes */
    printf("(%d) crc_32\n", cdh->crc_32);                          /* 4 bytes */
    printf("(%d) compressed_size\n", cdh->compressed_size);                 /* 4 bytes */
    printf("(%d) uncompressed_size\n", cdh->uncompressed_size);               /* 4 bytes */
    printf("(%d) file_name_length\n", cdh->file_name_length);                /* 2 bytes */
    printf("(%d) extra_field_length\n", cdh->extra_field_length);              /* 2 bytes */
    printf("(%d) file_comment_length\n", cdh->file_comment_length);             /* 2 bytes */
    printf("(%d) disk_number_start\n", cdh->disk_number_start);               /* 2 bytes */
    printf("(%d) internal_file_attributes\n", cdh->internal_file_attributes);        /* 2 bytes */
    printf("(%d) external_file_attributes\n", cdh->external_file_attributes);        /* 4 bytes */
    printf("(%d) relative_offset_of_local_header\n", cdh->relative_offset_of_local_header); /* 4 bytes */

    printf("Filename: %s\n", cdh->file_name); /* Not NULL terminated, so this is a bug */
    printf("Filecomment: %s\n", cdh->file_comment); /* Not NULL terminated, so this is a bug */
}

void _dump_local_fileheader(const local_file_header_t* lfh)
{
    printf("local_file_header_signature: %d\n", lfh->local_file_header_signature);
    printf("version_needed_to_extract:   %d\n", lfh->version_needed_to_extract);
    printf("general_purpose_bit_flag:    %d\n", lfh->general_purpose_bit_flag);
    printf("compression_method:          %d\n", lfh->compression_method);
    printf("last_mod_file_time:          %d\n", lfh->last_mod_file_time);
    printf("last_mod_file_date:          %d\n", lfh->last_mod_file_date);
    printf("crc_32:                      %d\n", lfh->crc_32);
    printf("compressed_size:             %d\n", lfh->compressed_size);
    printf("uncompressed_size:           %d\n", lfh->uncompressed_size);
    printf("file_name_length:            %d\n", lfh->file_name_length);
    printf("extra_field_length:          %d\n", lfh->extra_field_length);

    printf("Filename: %s\n", lfh->file_name);      /* Not NULL terminated, so this is a bug */
    printf("Extra field: %s\n", lfh->extra_field); /* Not NULL terminated, so this is a bug */
}
#endif  /* not NDEBUG */